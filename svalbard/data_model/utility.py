from types import UnionType
from typing import Any, Callable, Literal, Union, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from ..typing import TSettingValue
from .data_file import MetaData
from .instruments import (
    InstrumentModel,
    InstrumentSetting,
    SettingType,
    SweepCapability,
)


def _is_annotation_base_model(annotation: Any) -> bool:
    """Return True if annotation is a subclass or a union of BaseModel subclasses."""
    return issubclass(annotation, BaseModel)


def _is_literal(annotation: Any) -> bool:
    """Return True if annotation is a Literal."""
    return get_origin(annotation) is Literal


def _resolve_annotation(annotation: Any) -> SettingType:
    if get_origin(annotation) is Union or isinstance(annotation, UnionType):
        # field: Union[bool, int] or field: bool | int
        raise ValueError("Union types are not supported")
    if annotation is type(None):
        raise ValueError("None type is not supported")
    if get_origin(annotation) is list:
        settings_type = _resolve_annotation(get_args(annotation)[0])
        return SettingType(f"list[{settings_type.value}]")
    if _is_literal(annotation):
        return SettingType.STR
    if _is_annotation_base_model(annotation):
        return SettingType.MODEL
    return SettingType(str(annotation.__name__))


def _get_default_and_annotation(field_info: FieldInfo) -> tuple[Any, SettingType]:
    """Return the default value and the annotation of a Pydantic model field."""
    default = field_info.default
    if field_info.default is PydanticUndefined:
        default = None

    return default, _resolve_annotation(field_info.annotation)


def _instrument_setting_from_pydantic_field(
    setting_name: str, field_info: FieldInfo
) -> InstrumentSetting:
    """
    Create an InstrumentSetting from a Pydantic model field.

    Args:
        field_name (str): Name of the created setting.
        field_info (FieldInfo): The field information.

    Returns:
        InstrumentSetting: The created InstrumentSetting.
    """
    default, annotation = _get_default_and_annotation(field_info)
    return InstrumentSetting(
        name=setting_name,
        value=default,
        sweep_capability=SweepCapability.SW_GENERATED_HW_SET,
        dtype=SettingType(annotation),
    )


def instrument_model_from_pydantic_class(
    pydantic_class: type[BaseModel],
    instrument_identity: str,
    hardware: str,
) -> InstrumentModel:
    """
    Creates an InstrumentModel from a Pydantic class. A setting is created for each
    field in the Pydantic class. A setting is also created for the Pydantic class itself
    with the provided instrument_identity as the name and the Pydantic class name as the
    value.

    Args:
        pydantic_class (type[BaseModel]): The Pydantic class to create the model from.
        instrument_identity (str): Identity of the created instrument model.
        hardware (str): string to use as the hardware field in the InstrumentModel.

    Returns:
        InstrumentModel: created from the Pydantic class and the provided arguments.
    """
    settings = {}
    for field_name, field_info in pydantic_class.model_fields.items():
        settings[field_name] = _instrument_setting_from_pydantic_field(
            field_name, field_info
        )
    object_setting = InstrumentSetting(  # create Object setting
        name=instrument_identity,
        value=pydantic_class.__name__,
        sweep_capability=SweepCapability.NONE,
        dtype=SettingType.OBJECT,
    )
    settings[instrument_identity] = object_setting  # add Object setting
    return InstrumentModel(
        hardware=hardware,
        version="0.1",
        model=pydantic_class.__name__,
        identity=instrument_identity,
        settings=settings,
    )


def no_conversion(model: BaseModel) -> BaseModel:
    """Default model conversion function, returns the input model."""
    return model


def pydantic_model_from_metadata(
    metadata: MetaData,
    instrument_id: str,
    model_classes: dict[str, type[BaseModel]],
    converter: Callable[[BaseModel], BaseModel] = no_conversion,
) -> BaseModel:
    """Create a Pydantic model from metadata, instrument_id, and model_classes.

    Args:
        metadata (MetaData): The metadata to create the Pydantic model from.
        instrument_id (str): Identity of the InstrumentModel to create the model from.
        model_classes (dict[str, type[BaseModel]]):
            Dictionary of Pydantic model classes. Keyed by InstrumentModel.model field
            values. used to map the InstrumentModel to the correct Pydantic model.
        converter (Callable[[BaseModel], BaseModel], optional):
            User definable method to convert one Pydantic model into another. Called
            When submodels are created from the metadata. The default is no_conversion.

    Returns:
        BaseModel: Pydantic model instance created from the metadata.
    """

    def _submodel(submodel_id: str | TSettingValue) -> BaseModel:
        if not isinstance(submodel_id, str):
            raise ValueError(f"unsupported type ({type(submodel_id)}) for submodel id")
        pydantic_model = pydantic_model_from_metadata(
            metadata, submodel_id, model_classes, converter
        )
        return converter(pydantic_model)

    instrument_model = metadata.get_instrument_model(instrument_id)
    model_class = model_classes[instrument_model.model]
    field_values = {}
    for field_name, setting in instrument_model.settings.items():
        if field_name == instrument_id:  # remove the Object setting
            continue
        if setting.dtype == SettingType.LIST_MODEL and isinstance(setting.value, list):
            field_values[field_name] = [_submodel(value) for value in setting.value]
            continue
        if setting.dtype == SettingType.MODEL and isinstance(setting.value, str):
            field_values[field_name] = _submodel(setting.value)
            continue
        field_values[field_name] = setting.value
    return model_class(**field_values)
