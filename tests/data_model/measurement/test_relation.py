import json

import bson
import numpy as np
import pytest

from svalbard.data_model.measurement.lookup import LookupTable
from svalbard.data_model.measurement.relation import (
    RelationParameters,
    RelationSettings,
)
from svalbard.data_model.measurement.step_config import StepConfig


def test_relation_parameters():
    parameter = RelationParameters(variable="x", source_name="y")
    assert parameter.variable == "x"
    assert parameter.source_name == "y"
    assert not parameter.use_lookup
    assert parameter.lookup_table is None
    assert np.allclose(parameter.values(np.array([1, 2, 3])), np.array([1, 2, 3]))


def test_relation_parameters_bson():
    parameter = RelationParameters(variable="x", source_name="y")
    bson_encoded = bson.BSON.encode(json.loads(parameter.model_dump_json()))
    assert bson_encoded is not None
    bson_decoded = bson.BSON(bson_encoded).decode()
    new_parameter = RelationParameters(**bson_decoded)
    assert new_parameter == parameter


def test_relation_parameters_with_lookup():
    parameter = RelationParameters(
        variable="x",
        source_name="y",
        use_lookup=True,
        lookup_table=LookupTable.from_arrays(
            x=np.array([1, 2, 3]), y=np.array([4, 5, 6])
        ),
    )
    assert parameter.variable == "x"
    assert parameter.source_name == "y"
    assert parameter.use_lookup
    assert parameter.lookup_table is not None
    assert np.allclose(parameter.values(np.array([1, 2, 3])), np.array([4, 5, 6]))


def test_relation_parameters_with_lookup_bson():
    parameter = RelationParameters(
        variable="x",
        source_name="y",
        use_lookup=True,
        lookup_table=LookupTable.from_arrays(
            x=np.array([1, 2, 3]), y=np.array([4, 5, 6])
        ),
    )
    bson_encoded = bson.BSON.encode(json.loads(parameter.model_dump_json()))
    assert bson_encoded is not None
    bson_decoded = bson.BSON(bson_encoded).decode()
    new_parameter = RelationParameters(**bson_decoded)
    assert new_parameter == parameter


def test_relation_settings():
    relation = RelationSettings(
        name="rel1",
        config=StepConfig(),
        enable=True,
        parameters=[RelationParameters(variable="x", source_name="var1")],
    )
    assert relation.name == "rel1"
    assert relation.enable
    assert relation.equation == "x"
    assert len(relation.parameters) == 1

    values = relation.calculate_values({"var1": np.array([1, 2, 3])})
    assert np.allclose(values, np.array([1, 2, 3]))

    assert relation.dependency_names() == {"var1"}

    relation.set_equation("2 * x")
    assert relation.equation == "2 * x"
    values2 = relation.calculate_values({"var1": np.array([1, 2, 3])})
    assert np.allclose(values2, np.array([2, 4, 6]))

    relation.add_parameter(RelationParameters(variable="z", source_name="var2"))
    assert len(relation.parameters) == 2
    relation.set_equation("x + z")
    values3 = relation.calculate_values(
        {"var1": np.array([1, 2, 3]), "var2": np.array([4, 5, 6])}
    )
    assert np.allclose(values3, np.array([5, 7, 9]))

    bson_encoded = bson.BSON.encode(json.loads(relation.model_dump_json()))
    assert bson_encoded is not None
    bson_decoded = bson.BSON(bson_encoded).decode()
    new_relation = RelationSettings(**bson_decoded)
    assert new_relation == relation


def test_get_parameter():
    relation = RelationSettings(
        name="rel1",
        config=StepConfig(),
        enable=True,
        parameters=[RelationParameters(variable="x", source_name="var1")],
    )
    parameter = relation.get_parameter("x")
    assert parameter == relation.parameters[0]

    # invalid parameter name
    with pytest.raises(ValueError):
        relation.get_parameter("y")


def test_remove_parameter():
    relation = RelationSettings(
        name="rel1",
        config=StepConfig(),
        enable=True,
        parameters=[RelationParameters(variable="x", source_name="var1")],
    )
    assert relation.parameters
    relation.remove_parameter("x")
    assert not relation.parameters


def test_relation_param_name_pythonic():
    rel_param = RelationParameters(variable="x2", source_name="var 1")
    assert rel_param.source_name == "var_1"
