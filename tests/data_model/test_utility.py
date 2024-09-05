from types import NoneType
from typing import Literal, Union

import pytest
from pydantic import BaseModel

from svalbard.data_model._setting_types import SettingType
from svalbard.data_model.data_file import InstrumentModel, MetaData
from svalbard.data_model.utility import (
    _resolve_annotation,
    instrument_model_from_pydantic_class,
    pydantic_model_from_metadata,
)


class BasicModel(BaseModel):
    """A basic Pydantic model with a single field"""

    string_field: str
    int_field: int
    float_field: float
    bool_field: bool


class ComplexModel(BaseModel):
    """A more complex Pydantic model with a nested field"""

    basic_field: BasicModel
    list_field: list[BasicModel]
    string_field: str
    int_field: int
    float_field: float
    bool_field: bool


def test_to_instrument_model_basic():
    model = instrument_model_from_pydantic_class(BasicModel, "model_id", "hardware")
    assert model.hardware == "hardware"
    assert model.version == "0.1"
    assert model.model == "BasicModel"
    assert model.identity == "model_id"
    assert len(model.settings) == 5
    assert model.get_setting("model_id").dtype == SettingType.OBJECT
    assert model.get_setting("model_id").value == "BasicModel"
    assert model.get_setting("string_field").dtype == SettingType.STR
    assert model.get_setting("int_field").dtype == SettingType.INT
    assert model.get_setting("float_field").dtype == SettingType.FLOAT
    assert model.get_setting("bool_field").dtype == SettingType.BOOL


def test_to_instrument_model_complex():
    model = instrument_model_from_pydantic_class(ComplexModel, "model_id", "hardware")
    assert model.hardware == "hardware"
    assert model.version == "0.1"
    assert model.model == "ComplexModel"
    assert model.identity == "model_id"
    assert len(model.settings) == 7
    assert model.get_setting("model_id").dtype == SettingType.OBJECT
    assert model.get_setting("model_id").value == "ComplexModel"
    assert model.get_setting("basic_field").dtype == SettingType.MODEL
    assert model.get_setting("list_field").dtype == SettingType.LIST_MODEL
    assert model.get_setting("string_field").dtype == SettingType.STR
    assert model.get_setting("int_field").dtype == SettingType.INT
    assert model.get_setting("float_field").dtype == SettingType.FLOAT
    assert model.get_setting("bool_field").dtype == SettingType.BOOL


def basic_model_factory(index: int) -> InstrumentModel:
    model = instrument_model_from_pydantic_class(
        BasicModel, f"model_id_{index}", "hardware"
    )
    model.update_setting("string_field", f"string_{index}")
    model.update_setting("int_field", index)
    model.update_setting("float_field", float(index))
    model.update_setting("bool_field", bool(index % 2))
    return model


def test_from_instrument_model_basic(metadata: MetaData):
    basic_model = basic_model_factory(1)
    metadata.add_instrument_model(basic_model)
    pydantic_model = pydantic_model_from_metadata(
        metadata, basic_model.identity, {"BasicModel": BasicModel}
    )
    assert isinstance(pydantic_model, BasicModel)
    assert pydantic_model.string_field == "string_1"
    assert pydantic_model.int_field == 1
    assert pydantic_model.float_field == 1.0
    assert pydantic_model.bool_field is True


def test_from_instrument_model_complex(metadata: MetaData):
    complex_model = instrument_model_from_pydantic_class(
        ComplexModel, "complex_model_id", "hardware"
    )
    assert complex_model.get_setting("basic_field").dtype == SettingType.MODEL
    assert complex_model.get_setting("list_field").dtype == SettingType.LIST_MODEL
    metadata.add_instrument_model(complex_model)
    complex_model.update_setting("list_field", [])
    for index in range(4):
        basic_model = basic_model_factory(index)
        metadata.add_instrument_model(basic_model)
        if index == 0:
            complex_model.update_setting("basic_field", basic_model.identity)
        else:
            assert isinstance(complex_model.settings["list_field"].value, list)
            complex_model.settings["list_field"].value.append(basic_model.identity)
    complex_model.update_setting("basic_field", "model_id_0")
    complex_model.update_setting(
        "list_field",
        [
            "model_id_1",
            "model_id_2",
            "model_id_3",
        ],
    )
    complex_model.update_setting("string_field", "string")
    complex_model.update_setting("int_field", 1)
    complex_model.update_setting("float_field", 1.0)
    complex_model.update_setting("bool_field", True)
    pydantic_model = pydantic_model_from_metadata(
        metadata,
        complex_model.identity,
        {"ComplexModel": ComplexModel, "BasicModel": BasicModel},
    )
    assert isinstance(pydantic_model, ComplexModel)
    assert pydantic_model.string_field == "string"
    assert pydantic_model.int_field == 1
    assert pydantic_model.float_field == 1.0
    assert pydantic_model.bool_field is True

    assert isinstance(pydantic_model.basic_field, BasicModel)
    assert pydantic_model.basic_field.string_field == "string_0"
    assert pydantic_model.basic_field.int_field == 0
    assert pydantic_model.basic_field.float_field == 0.0
    assert pydantic_model.basic_field.bool_field is False

    assert len(pydantic_model.list_field) == 3
    for index, basic_field in enumerate(pydantic_model.list_field):
        assert isinstance(basic_field, BasicModel)
        assert basic_field.string_field == f"string_{index + 1}"
        assert basic_field.int_field == index + 1
        assert basic_field.float_field == float(index + 1)
        assert basic_field.bool_field is bool((index + 1) % 2)


def test_from_instrument_model_complex_errors(metadata: MetaData):
    complex_model = instrument_model_from_pydantic_class(
        ComplexModel, "complex_model_id", "hardware"
    )
    metadata.add_instrument_model(complex_model)
    complex_model.update_setting("basic_field", 32)

    with pytest.raises(ValueError):
        # Values of model fields must be strings
        pydantic_model_from_metadata(
            metadata,
            complex_model.identity,
            {"ComplexModel": ComplexModel, "BasicModel": BasicModel},
        )

    basic_model = basic_model_factory(0)
    metadata.add_instrument_model(basic_model)
    complex_model.update_setting("basic_field", basic_model.identity)
    complex_model.update_setting("list_field", [32, 12])

    with pytest.raises(ValueError):
        # Values of list model fields must be strings
        pydantic_model_from_metadata(
            metadata,
            complex_model.identity,
            {"ComplexModel": ComplexModel, "BasicModel": BasicModel},
        )


class ListModel(BaseModel):
    list_field_str: list[str]
    list_field_int: list[int]
    list_field_float: list[float]
    list_field_bool: list[bool]


def test_to_instrument_model_list():
    model = instrument_model_from_pydantic_class(ListModel, "model_id", "hardware")
    assert model.hardware == "hardware"
    assert model.version == "0.1"
    assert model.model == "ListModel"
    assert model.identity == "model_id"
    assert len(model.settings) == 5
    assert model.get_setting("model_id").dtype == SettingType.OBJECT
    assert model.get_setting("model_id").value == "ListModel"
    assert model.get_setting("list_field_str").dtype == SettingType.LIST_STR
    assert model.get_setting("list_field_int").dtype == SettingType.LIST_INT
    assert model.get_setting("list_field_float").dtype == SettingType.LIST_FLOAT
    assert model.get_setting("list_field_bool").dtype == SettingType.LIST_BOOL


def test_from_instrument_model_list(metadata: MetaData):
    model = instrument_model_from_pydantic_class(ListModel, "model_id", "hardware")
    model.update_setting("list_field_str", ["string_1", "string_2"])
    model.update_setting("list_field_int", [1, 2])
    model.update_setting("list_field_float", [1.0, 2.0])
    model.update_setting("list_field_bool", [True, False])
    metadata.add_instrument_model(model)
    pydantic_model = pydantic_model_from_metadata(
        metadata, model.identity, {"ListModel": ListModel}
    )
    assert isinstance(pydantic_model, ListModel)
    assert pydantic_model.list_field_str == ["string_1", "string_2"]
    assert pydantic_model.list_field_int == [1, 2]
    assert pydantic_model.list_field_float == [1.0, 2.0]
    assert pydantic_model.list_field_bool == [True, False]


def test_model_values_must_be_assigned(metadata: MetaData):
    model = instrument_model_from_pydantic_class(ListModel, "model_id", "hardware")
    metadata.add_instrument_model(model)
    with pytest.raises(ValueError):
        pydantic_model_from_metadata(metadata, model.identity, {"ListModel": ListModel})


def test_conversion(metadata):
    class Model1(BaseModel):
        field: str

    class Model2(BaseModel):
        field: str

    class Model1Generator(BaseModel):
        field: str

        def to_model1(self):
            return Model1(field=self.field)

    def conversion_func(model: BaseModel) -> BaseModel:
        if isinstance(model, Model1Generator):
            return model.to_model1()
        return model

    class BigModel(Model1Generator):
        model1: Model1
        model2: Model2

        def to_model1(self):
            return Model1(field=self.model1.field + self.model2.field + self.field)

    model1 = instrument_model_from_pydantic_class(
        Model1Generator, "model1", "hardware1"
    )
    model1.update_setting("field", "string1")
    model2 = instrument_model_from_pydantic_class(Model2, "model2", "hardware2")
    model2.update_setting("field", "string2")
    big_model = instrument_model_from_pydantic_class(BigModel, "model_id", "hardware_u")
    big_model.update_setting("model1", "model1")
    big_model.update_setting("model2", "model2")
    big_model.update_setting("field", "string3")

    metadata.add_instrument_model(model1)
    metadata.add_instrument_model(model2)
    metadata.add_instrument_model(big_model)

    models = {
        "Model1": Model1,
        "Model2": Model2,
        "BigModel": BigModel,
        "Model1Generator": Model1Generator,
    }

    pydantic_model = pydantic_model_from_metadata(
        metadata, big_model.identity, models, conversion_func
    )

    assert isinstance(pydantic_model, BigModel)
    assert isinstance(pydantic_model.model1, Model1)
    assert pydantic_model.model1.field == "string1"
    assert isinstance(pydantic_model.model2, Model2)
    assert pydantic_model.model2.field == "string2"

    assert pydantic_model.field == "string3"
    assert pydantic_model.to_model1().field == "string1string2string3"


def test_literal_field():
    class LiteralModel(BaseModel):
        literal_field: Literal["string1", "string2"]

    instr = instrument_model_from_pydantic_class(LiteralModel, "model_id", "hardware")
    assert instr.hardware == "hardware"
    instr.update_setting("literal_field", "string1")
    metadata = MetaData()
    metadata.add_instrument_model(instr)

    model = pydantic_model_from_metadata(
        metadata, instr.identity, {"LiteralModel": LiteralModel}
    )
    assert isinstance(model, LiteralModel)
    assert model.literal_field == "string1"


def test_resolve_annotation():
    assert _resolve_annotation(str) == SettingType.STR
    assert _resolve_annotation(int) == SettingType.INT
    assert _resolve_annotation(float) == SettingType.FLOAT
    assert _resolve_annotation(bool) == SettingType.BOOL
    assert _resolve_annotation(list[str]) == SettingType.LIST_STR
    assert _resolve_annotation(list[int]) == SettingType.LIST_INT
    assert _resolve_annotation(list[float]) == SettingType.LIST_FLOAT
    assert _resolve_annotation(list[bool]) == SettingType.LIST_BOOL
    assert _resolve_annotation(Literal["string1", "string2"]) == SettingType.STR
    assert _resolve_annotation(BasicModel) == SettingType.MODEL
    assert _resolve_annotation(list[BasicModel]) == SettingType.LIST_MODEL
    with pytest.raises(ValueError):
        _resolve_annotation(Union[str, int])
    with pytest.raises(ValueError):
        _resolve_annotation(NoneType)
