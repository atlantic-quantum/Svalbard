"""
Module for handling relations between channels, i.e. channels whose values are
dependent on other channels.
"""

import asteval
import numpy as np
from pydantic import BaseModel, Field, validator

from .lookup import LookupTable
from .step_config import StepConfig


# how to handle relations with multiple variables?
class RelationParameters(BaseModel):
    """
    Pydantic model for relation parameters,

    Args:
        variable (str):
            variable name of parameter, should be a valid asteval symbol name
            used to identify the parameter in the equation for the relation settings
        source_name (str):
            name of source channel for this parameter,
        use_lookup (bool):
            whether to use a lookup table to calculate values for this parameter
            default False
        lookup_table (LookupTable):
            lookup table to use to calculate values for this parameter
            if use_lookup is True, default None

    """

    variable: str  # variable in equation this parameter is for
    source_name: str  # source channel name
    use_lookup: bool = False
    lookup_table: LookupTable | None = None

    @validator("variable")
    def variable_must_be_valid_symbol_name(cls, v: str):
        """Validate that variable is a valid asteval symbol name"""
        assert asteval.valid_symbol_name(v)
        return v

    def values(self, source_values: np.ndarray) -> np.ndarray:
        """
        calculate values for this parameter from source values, if use_lookup is True
        the lookup table is used to calculate the values otherwise the source values
        are returned

        Args:
            source_values (np.ndarray): values of source channel

        Returns:
            np.ndarray: values for this parameter
        """
        if self.use_lookup:
            assert self.lookup_table is not None
            return self.lookup_table.calculate_values(source_values)
        return source_values


class RelationSettings(BaseModel):
    """
    Pydantic model for relation settings, i.e. settings for a relation between channels

    Args:
        name (str):
            name of channel this relation is applied to
        config (StepConfig):
            step config for this relation, i.e. how the relation is applied
        enable (bool):
            whether this relation is enabled, default False
        equation (str):
            right hand side of equation to calculate values for this channel,
            default "x"
        parameters (list[RelationParameters]):
            list of parameters for this relation, default empty list
    """

    name: str  # what (channel) the relation is applied to
    config: StepConfig  # how the relation is applied
    enable: bool = False  # keep for capability of toggling relations on/off
    equation: str = "x"  # equation to calculate values for this channel
    parameters: list[RelationParameters] = Field(default_factory=list)

    def calculate_values(self, source_values_dict: dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculate values for this relation from a dictionary of source values,
        each entry in the dictionary should be a numpy array with the values for
        a source channel corresponding to the source name of a relation parameter
        in these relation settings

        Args:
            source_values_dict (dict[str, np.ndarray]): dictionary of source values

        Returns:
            np.ndarray: values for this relation calculated from source values
        """
        parameter_values = {
            parameter.variable: parameter.values(
                source_values_dict[parameter.source_name]
            )
            for parameter in self.parameters
        }
        symbol_table = asteval.make_symbol_table(use_numpy=True, **parameter_values)
        aeval = asteval.Interpreter(symtable=symbol_table)
        # todo should we allow parameter named result?
        aeval(f"result = {self.equation}")
        result = aeval.symtable["result"]
        assert isinstance(result, np.ndarray)
        return result

    def add_parameter(self, parameter: RelationParameters):
        """
        Add a parameter to these relation settings

        Args:
            parameter (RelationParameters): parameter to add
        """
        self.parameters.append(parameter)

    def get_parameter(self, name: str) -> RelationParameters:
        """
        get relation parameter by name of variable

        Args:
            name (str):
                variable name of parameter

        Raises:
            ValueError:
                if no parameter with given name is found in these relation settings

        Returns:
            RelationParameters: the parameter with given variable name
        """
        for p in self.parameters:
            if p.variable == name:
                return p
        raise ValueError(f"Parameter {name} not found")

    def remove_parameter(self, name: str):
        """
        Remove parameter with given variable name from these relation settings

        Args:
            name (str): variable name of parameter to remove
        """
        parameter = self.get_parameter(name)
        self.parameters.remove(parameter)

    def set_equation(self, equation: str):
        """
        Set equation for these relation settings, should be the expression to calculate,
        i.e. the right hand side of the equation

        Args:
            equation (str): equation to set
        """
        self.equation = equation

    def dependency_names(self) -> set[str]:
        """
        Get the names of the source channels for these relation settings,
        i.e. the names in the source_name field of the relation parameters and
        used to identify source values in the calculate_values method

        Returns:
            set[str]: _description_
        """
        return {p.source_name for p in self.parameters}
