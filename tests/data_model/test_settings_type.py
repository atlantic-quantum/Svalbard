import enum

import pytest

import svalbard.data_model._setting_types as setting_types


def test_list_setting_type_factory():
    assert issubclass(setting_types.ListFloatSettingType, setting_types.BaseSettingType)
    assert setting_types.ListFloatSettingType.__name__ == "ListFloatSettingType"
    assert (
        setting_types.ListFloatSettingType
        in setting_types.BaseSettingType.__subclasses__()
    )
    assert (
        setting_types.ListFloatSettingType().setting_type
        == setting_types.SettingType.LIST_FLOAT
    )


setting_types_list = [
    (setting_types.SettingType.FLOAT, setting_types.FloatSettingType),
    (setting_types.SettingType.INT, setting_types.IntSettingType),
    (setting_types.SettingType.BOOL, setting_types.BoolSettingType),
    (setting_types.SettingType.ENUM, setting_types.EnumSettingType),
    (setting_types.SettingType.STR, setting_types.StrSettingType),
    (setting_types.SettingType.COMPLEX, setting_types.ComplexSettingType),
    (setting_types.SettingType.LIST_STR, setting_types.ListStrSettingType),
    (setting_types.SettingType.LIST_INT, setting_types.ListIntSettingType),
    (setting_types.SettingType.LIST_BOOL, setting_types.ListBoolSettingType),
    (setting_types.SettingType.LIST_ENUM, setting_types.ListEnumSettingType),
    (setting_types.SettingType.LIST_FLOAT, setting_types.ListFloatSettingType),
    (setting_types.SettingType.LIST_COMPLEX, setting_types.ListComplexSettingType),
    (setting_types.SettingType.OBJECT, setting_types.ObjectSettingType),
    (setting_types.SettingType.NONE, setting_types.NoneSettingType),
    (setting_types.SettingType.LIST_MODEL, setting_types.ListModelSettingType),
    (setting_types.SettingType.MODEL, setting_types.ModelSettingType),
]


@pytest.mark.parametrize(("setting_type", "type_class"), setting_types_list)
def test_get_setting_type_class(
    setting_type: setting_types.SettingType,
    type_class: type[setting_types.BaseSettingType],
):
    assert setting_types.SettingType.get_setting_type_class(setting_type) == type_class


class SomeEnum(enum.Enum):
    A = "A"
    B = "B"


setting_types_to_value = [
    (setting_types.SettingType.FLOAT, 1.0, 1.0),
    (setting_types.SettingType.FLOAT, 1, 1.0),
    (setting_types.SettingType.INT, 1, 1),
    (setting_types.SettingType.INT, 1.0, 1),
    (setting_types.SettingType.BOOL, True, True),
    (setting_types.SettingType.BOOL, 0, False),
    (setting_types.SettingType.BOOL, 1, True),
    (setting_types.SettingType.ENUM, SomeEnum.A, "A"),
    (setting_types.SettingType.STR, "A", "A"),
    (setting_types.SettingType.COMPLEX, "1+1j", "1+1j"),
    (setting_types.SettingType.LIST_STR, ["A"], ["A"]),
    (setting_types.SettingType.LIST_INT, [1], [1]),
    (setting_types.SettingType.LIST_BOOL, [True], [True]),
    (setting_types.SettingType.LIST_ENUM, ["A"], ["A"]),
    (setting_types.SettingType.LIST_FLOAT, [1.0], [1.0]),
    (setting_types.SettingType.LIST_COMPLEX, ["1+1j"], ["1+1j"]),
    (setting_types.SettingType.LIST_COMPLEX, [1 + 1j], ["1.0+1.0j"]),
    (setting_types.SettingType.LIST_COMPLEX, ["1-1j"], ["1-1j"]),
    (setting_types.SettingType.LIST_COMPLEX, [1 - 1j], ["1.0-1.0j"]),
    (setting_types.SettingType.OBJECT, "A", "A"),
    (setting_types.SettingType.NONE, None, None),
    (setting_types.SettingType.NONE, [], []),
    (setting_types.SettingType.LIST_MODEL, ["a", "b"], ["a", "b"]),
    (setting_types.SettingType.MODEL, "A", "A"),
]


@pytest.mark.parametrize(
    ("setting_type", "input_value", "expected"), setting_types_to_value
)
def test_setting_type_to_value(
    setting_type: setting_types.SettingType, input_value, expected
):
    type_class = setting_types.SettingType.get_setting_type_class(setting_type)
    assert type_class.to_value(input_value) == expected


def test_complex_to_str_error():
    with pytest.raises(ValueError):
        setting_types.ComplexSettingType.to_value("not a complex")


def test_none_to_value_error():
    with pytest.raises(ValueError):
        setting_types.NoneSettingType.to_value(1)  # type: ignore

    with pytest.raises(ValueError):
        setting_types.NoneSettingType.to_value([1])


determine_setting_type_list = [
    # setting type is None, expected type is the same as the value type
    (1.0, None, setting_types.SettingType.FLOAT),
    (1, None, setting_types.SettingType.INT),
    (True, None, setting_types.SettingType.BOOL),
    ("A", None, setting_types.SettingType.STR),
    (1 + 1j, None, setting_types.SettingType.COMPLEX),
    (["A"], None, setting_types.SettingType.LIST_STR),
    ([1], None, setting_types.SettingType.LIST_INT),
    ([True], None, setting_types.SettingType.LIST_BOOL),
    ([SomeEnum.A], None, setting_types.SettingType.LIST_ENUM),
    ([1.0], None, setting_types.SettingType.LIST_FLOAT),
    ([1 + 1j], None, setting_types.SettingType.LIST_COMPLEX),
    # setting type is given, expected type is the same as the setting type
    ("A", setting_types.SettingType.FLOAT, setting_types.SettingType.FLOAT),
    (1, setting_types.SettingType.FLOAT, setting_types.SettingType.FLOAT),
    (1.0, setting_types.SettingType.INT, setting_types.SettingType.INT),
    (1, setting_types.SettingType.INT, setting_types.SettingType.INT),
    (True, setting_types.SettingType.BOOL, setting_types.SettingType.BOOL),
    (0, setting_types.SettingType.BOOL, setting_types.SettingType.BOOL),
    (1, setting_types.SettingType.BOOL, setting_types.SettingType.BOOL),
    (SomeEnum.A, setting_types.SettingType.ENUM, setting_types.SettingType.ENUM),
    ("A", setting_types.SettingType.STR, setting_types.SettingType.STR),
    ("1+1j", setting_types.SettingType.COMPLEX, setting_types.SettingType.COMPLEX),
    (["A"], setting_types.SettingType.LIST_STR, setting_types.SettingType.LIST_STR),
    ([1], setting_types.SettingType.LIST_INT, setting_types.SettingType.LIST_INT),
    ([True], setting_types.SettingType.LIST_BOOL, setting_types.SettingType.LIST_BOOL),
    (
        [SomeEnum.A],
        setting_types.SettingType.LIST_ENUM,
        setting_types.SettingType.LIST_ENUM,
    ),
    ([1.0], setting_types.SettingType.LIST_FLOAT, setting_types.SettingType.LIST_FLOAT),
    (
        [1 + 1j],
        setting_types.SettingType.LIST_COMPLEX,
        setting_types.SettingType.LIST_COMPLEX,
    ),
    ("a", setting_types.SettingType.OBJECT, setting_types.SettingType.OBJECT),
    ("a", setting_types.SettingType.MODEL, setting_types.SettingType.MODEL),
    (
        ["a", "b"],
        setting_types.SettingType.LIST_MODEL,
        setting_types.SettingType.LIST_MODEL,
    ),
    # setting type is a string, expected type is the same as the setting type
    ("A", "float", setting_types.SettingType.FLOAT),
    (1, "float", setting_types.SettingType.FLOAT),
    (1.0, "int", setting_types.SettingType.INT),
    (1, "int", setting_types.SettingType.INT),
    (True, "bool", setting_types.SettingType.BOOL),
    (0, "bool", setting_types.SettingType.BOOL),
    (1, "bool", setting_types.SettingType.BOOL),
    (SomeEnum.A, "enum", setting_types.SettingType.ENUM),
    ("A", "str", setting_types.SettingType.STR),
    ("1+1j", "complex", setting_types.SettingType.COMPLEX),
    (["A"], "list[str]", setting_types.SettingType.LIST_STR),
    ([1], "list[int]", setting_types.SettingType.LIST_INT),
    ([True], "list[bool]", setting_types.SettingType.LIST_BOOL),
    ([SomeEnum.A], "list[enum]", setting_types.SettingType.LIST_ENUM),
    ([1.0], "list[float]", setting_types.SettingType.LIST_FLOAT),
    ([1 + 1j], "list[complex]", setting_types.SettingType.LIST_COMPLEX),
    # setting type is SettingType.NONE, expected type is the same as the value type
    (None, setting_types.SettingType.NONE, setting_types.SettingType.NONE),
    ([], setting_types.SettingType.NONE, setting_types.SettingType.NONE),
    (1.0, setting_types.SettingType.NONE, setting_types.SettingType.FLOAT),
    (1, setting_types.SettingType.NONE, setting_types.SettingType.INT),
    (True, setting_types.SettingType.NONE, setting_types.SettingType.BOOL),
    ("A", setting_types.SettingType.NONE, setting_types.SettingType.STR),
    (1 + 1j, setting_types.SettingType.NONE, setting_types.SettingType.COMPLEX),
    (["A"], setting_types.SettingType.NONE, setting_types.SettingType.LIST_STR),
    ([1], setting_types.SettingType.NONE, setting_types.SettingType.LIST_INT),
    ([True], setting_types.SettingType.NONE, setting_types.SettingType.LIST_BOOL),
    ([SomeEnum.A], setting_types.SettingType.NONE, setting_types.SettingType.LIST_ENUM),
    ([1.0], setting_types.SettingType.NONE, setting_types.SettingType.LIST_FLOAT),
    ([1 + 1j], setting_types.SettingType.NONE, setting_types.SettingType.LIST_COMPLEX),
]


@pytest.mark.parametrize(
    ("value", "setting_type", "expected_type"), determine_setting_type_list
)
def test_determine_setting_type(
    value,
    setting_type: setting_types.SettingType | str | None,
    expected_type: setting_types.SettingType,
):
    assert (
        setting_types.SettingType.determine_setting_type(value, setting_type)
        == expected_type
    )


def test_determine_setting_type_invalid():
    with pytest.raises(ValueError):
        setting_types.SettingType.determine_setting_type(1, ["invalid"])  # type: ignore


get_value_string_list = [
    (setting_types.FloatSettingType, 1.0, "V", "1 V"),
    (setting_types.IntSettingType, 1, "V", "1 V"),
    (setting_types.BoolSettingType, True, "", "On"),
    (setting_types.BoolSettingType, False, "", "Off"),
    (setting_types.EnumSettingType, SomeEnum.A, "", "A"),
    (setting_types.EnumSettingType, "A", "", "A"),
    (setting_types.EnumSettingType, 1, "", "1"),
    (setting_types.StrSettingType, "A", "", "A"),
    (setting_types.ComplexSettingType, 1 + 1j, "", "1.41421 < 45.0°"),
    (setting_types.ComplexSettingType, "1+1j", "", "1.41421 < 45.0°"),
    (setting_types.ListStrSettingType, ["A"], "", "['A']"),
    (setting_types.ListIntSettingType, [1, 2], "", "['1', '2']"),
    (setting_types.ListBoolSettingType, [True, False], "", "['On', 'Off']"),
    (setting_types.ListEnumSettingType, [SomeEnum.A], "", "['A']"),
    (setting_types.ListFloatSettingType, [1.0, 2.0], "", "['1', '2']"),
    (
        setting_types.ListComplexSettingType,
        [1 + 1j, 1 - 1j],
        "",
        "['1.41421 < 45.0°', '1.41421 < -45.0°']",
    ),
    (setting_types.ObjectSettingType, "A", "", "A"),
    (setting_types.NoneSettingType, None, "", ""),
    (setting_types.ModelSettingType, "A", "", "A"),
    (setting_types.ListModelSettingType, ["a", "b"], "", "['a', 'b']"),
]


@pytest.mark.parametrize(
    ("type_class", "value", "unit", "expected"), get_value_string_list
)
def test_get_value_string(
    type_class: setting_types.BaseSettingType, value, unit: str, expected: str
):
    assert type_class.get_value_string(value, unit) == expected
