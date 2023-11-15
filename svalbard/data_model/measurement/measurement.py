"""
Data model for a measurement.
A measurement model is a collection of
  - channels used in a measurements.
  - step items that indicate how to step a subset of the channels,
  - relations that indicate how to set a different subset of the channels based on
    the values of the step items.
  - log channels that indicate which channels to log during the measurement.

"""

import numpy as np
import rustworkx as rx
from pydantic import BaseModel, validator

from .channel import Channel
from .relation import RelationSettings
from .step_config import StepConfig
from .step_item import StepItem
from .step_range import InterpolationTypes, RangeTypes, StepRange, StepTypes


class Measurement(BaseModel):
    """
    Pydantic model for a measurement, a measurement is a collection of:
      - channels used in a measurements.
      - step items that indicate how to step a subset of the channels,
      - relations that indicate how to set a different subset of the channels based on
        the values of the step items.
      - log channels that indicate which channels to log during the measurement.

    Args:
        channels (list[Channel]):
            List of channels used in the measurement.
        step_items (list[StepItem]):
            List of StepItems that indicate how to step a subset of the channels.
        relations (list[RelationSettings]):
            List of RelationSettings that indicate how to set a different subset
            of the channels, during the measurement, based on the values of the
            step items.
        log_channels (list[str]):
            List of channel names that indicate which channels to log during the
            Measurement.

    """

    channels: list[Channel]
    step_items: list[StepItem]
    relations: list[RelationSettings]
    log_channels: list[str]

    @validator("relations")
    def validate_relations_names_unique(cls, v: list[RelationSettings]):
        """validate that all relation names are unique"""
        names = {relation.name for relation in v}
        assert len(names) == len(v)
        return v

    @validator("relations")
    def validate_relations_names_exist(cls, v: list[RelationSettings], values: dict):
        """validate that all relation names exist as channels"""
        assert "channels" in values
        channel_names = {channel.name for channel in values["channels"]}
        for relation in v:
            assert relation.name in channel_names
        return v

    @validator("step_items")
    def validate_step_items_names_unique(cls, v: list[StepItem]):
        """validate that all step item names are unique"""
        names = {step.name for step in v}
        assert len(names) == len(v)
        return v

    @validator("step_items")
    def validate_step_items_names_exist(cls, v: list[StepItem], values: dict):
        """validate that all step item names exist as channels"""
        assert "channels" in values
        channel_names = {channel.name for channel in values["channels"]}
        for step in v:
            assert step.name in channel_names
        return v

    def add_step(
        self,
        channel: str | Channel,
        values: np.ndarray | list[float] | float,
        index: int,
    ):
        """
        Adds a step item to the list of step items by specifying a channel name
        (or Channel object), a list of values and an index. A new step item is created
        and inserted at the given index in the list of step items.

        Args:
            channel (str | Channel):
                name of the channel (or Channel object) to add a step item for
            values (np.ndarray | list[float] | float):
                values to step through in the step item
            index (int):
                index to insert the new step item at

        Raises:
            ValueError:
                if the channel name does not exist in the list of channels
        """
        if isinstance(channel, str):
            if channel not in self.channel_names:
                raise ValueError(f"Channel {channel} does not exist")
            for c in self.channels:
                if c.name == channel:
                    channel = c
                    break
        else:
            if channel.name not in self.channel_names:
                self.channels.append(channel)
        if isinstance(values, float):
            values = [values]
        if isinstance(values, np.ndarray):
            values = values.tolist()
        assert isinstance(values, list)
        assert isinstance(channel, Channel)

        step_item = StepItem(
            name=channel.name,
            config=StepConfig(),
            ranges=[
                StepRange(
                    range_type=RangeTypes.VALUES,
                    step_type=StepTypes.STEP_COUNT,
                    interpolation_type=InterpolationTypes.LINEAR,
                    values=values,
                )
            ],
        )

        self.step_items.insert(index, step_item)

    @property
    def relation_names(self) -> list[str]:
        """return set of relation names"""
        return [relation.name for relation in self.relations]

    @property
    def step_item_names(self) -> list[str]:
        """return set of step item names"""
        return [step.name for step in self.step_items]

    @property
    def channel_names(self) -> list[str]:
        """return set of channel names"""
        return [channel.name for channel in self.channels]

    @property
    def log_shape(self) -> tuple[int, ...]:
        """Returns the shape of the log data"""
        return tuple(step.calculate_values().size for step in self.step_items)

    def add_log(self, name: str):
        """
        Add a channel to the list of channels to log

        Args:
            name (str):
                name of the channel to add

        Raises:
            ValueError:
                if no channel with the given name exists in the Channel list
        """
        if name not in self.channel_names:
            raise ValueError(f"Channel {name} does not exist")
        self.log_channels.append(name)

    def add_relation(self, relation: RelationSettings):
        """
        Add a RelationSettings to the list of relations

        Args:
            relation (RelationSettings):
                the relation to add

        Raises:
            ValueError:
                if a relation with the same name already exists
            ValueError:
                if no channel with the given name exists in the Channel list
        """
        if relation.name in self.relation_names:
            raise ValueError(f"Relation for channel {relation.name} already exists")
        if relation.name not in self.channel_names:
            raise ValueError(f"Channel {relation.name} does not exist")
        self.relations.append(relation)

    def add_step_item(self, step_item: StepItem, index: int | None = None):
        """
        Add a StepItem to the list of step items

        Args:
            step_item (StepItem):
                step item to add
            index (int | None, optional):
                where to insert the new step item in the step item list.
                Defaults to None. If None the step item is appended to the end of list.

        Raises:
            ValueError:
                if a step item with the same name already exists
            ValueError:
                if no channel with the given name exists in the Channel list
        """
        if step_item.name in self.step_item_names:
            raise ValueError(f"Step item for channel {step_item.name} already exists")
        if step_item.name not in self.channel_names:
            raise ValueError(f"Channel {step_item.name} does not exist")
        if index is None:
            self.step_items.append(step_item)
        else:
            self.step_items.insert(index, step_item)

    def add_channel(self, channel: Channel):
        """
        Add a channel to the list of channels

        Args:
            channel (Channel):
                Channel to add

        Raises:
            ValueError:
                if a channel with the same name already exists
        """
        if channel.name in self.channel_names:
            raise ValueError(f"Channel name {channel.name} already exists")
        self.channels.append(channel)

    def get_channel(self, name: str) -> Channel:
        """
        returns a channel object by name

        Args:
            name (str):
                name of the channel to get

        Raises:
            ValueError:
                if no channel with the given name exists in the Channel list

        Returns:
            Channel: channel object with the given name
        """
        for channel in self.channels:
            if channel.name == name:
                return channel
        raise ValueError(f"Channel {name} does not exist")

    def get_step_item(self, name: str) -> StepItem:
        """
        returns a StepItem object by name

        Args:
            name (str):
                name of the step item to get

        Raises:
            ValueError:
                if no step item with the given name exists in the step item list

        Returns:
            StepItem: step item object with the given name
        """
        for step_item in self.step_items:
            if step_item.name == name:
                return step_item
        raise ValueError(f"Step item {name} does not exist")

    def get_relation(self, name: str) -> RelationSettings:
        """
        Return a RelationSetting object by name

        Args:
            name (str):
                name of the relation to get

        Raises:
            ValueError:
                if no relation with the given name exists in the relation list

        Returns:
            RelationSettings: relation object with the given name
        """
        for relation in self.relations:
            if relation.name == name:
                return relation
        raise ValueError(f"Relation {name} does not exist")

    def remove_channel(self, name: str):
        """
        Remove a channel from the list of channels by name

        Args:
            name (str):
                name of the channel to remove
        """
        channel = self.get_channel(name)
        self.channels.remove(channel)

    def remove_step_item(self, name: str):
        """
        Remove a step item from the list of step items by name

        Args:
            name (str):
                name of the step item to remove
        """
        step_item = self.get_step_item(name)
        self.step_items.remove(step_item)

    def remove_relation(self, name: str):
        """
        Remove a relation from the list of relations by name

        Args:
            name (str):
                name of the relation to remove
        """
        relation = self.get_relation(name)
        self.relations.remove(relation)

    def remove_log(self, name: str):
        """
        Remove a log channel from the list of log channels by name

        Args:
            name (str):
                name of the log channel to remove
        """
        self.log_channels.remove(name)

    def set_log_position(self, name: str, index: int):
        """
        Change the position of a log channel in the list of log channels

        Args:
            name (str):
                name of the log channel to move
            index (int):
                new index of the log channel
        """
        self.log_channels.remove(name)
        self.log_channels.insert(index, name)

    def set_step_item_position(self, name: str, index: int):
        """
        Change the position of a step item in the list of step items

        Args:
            name (str):
                name of the step item to move
            index (int):
                new index of the step item
        """
        step_item = self.get_step_item(name)
        self.step_items.remove(step_item)
        self.step_items.insert(index, step_item)

    def calculate_values(self, expand_arrays: bool = True) -> dict[str, np.ndarray]:
        """
        Calculate values for all step items and relations in the measurement.

        Returns:
            dict[str, np.ndarray]:
                dictionary of all step item and relation values
                with the channel name as key and the values as numpy array
            # Todo Docstring
            # Todo better name than expand_arrays
        """
        step_values = {step.name: step.calculate_values() for step in self.step_items}
        step_values = step_arrays(step_values) if expand_arrays else step_values

        dependency_graph = self.create_relations_dependency_graph()
        relation_dict = {relation.name: relation for relation in self.relations}

        for node_index in rx.topological_sort(dependency_graph):  # type: ignore
            name = dependency_graph.nodes()[node_index]
            if name in relation_dict:
                if relation_dict[name].enable:
                    step_values[name] = relation_dict[name].calculate_values(
                        step_values
                    )

        return step_values

    def create_relations_dependency_graph(self) -> rx.PyDiGraph:
        """
        Create a dependency graph for the relations and step items in the measurement,
        can be used to determine the order in which to calculate the values of the
        relations. or to draw a graph of the relations.

        Raises:
            ValueError: if a relation depends on a non-existing step item

        Returns:
            rx.PyDiGraph: Directed acyclic graph of the relations and step items
        """
        graph = rx.PyDiGraph()
        for step in self.step_items:
            graph.add_node(step.name)
        for relation in self.relations:
            graph.add_node(relation.name)
        for relation in self.relations:
            for dependency in relation.dependency_names():
                try:
                    graph.add_edge(
                        graph.nodes().index(dependency),
                        graph.nodes().index(relation.name),
                        f"{dependency} -> {relation.name}",
                    )
                except ValueError as e:
                    estr = f"Relation {relation.name} depends on"
                    estr += f" non-existing step item {dependency}"
                    raise ValueError(estr) from e

        assert rx.is_directed_acyclic_graph(graph)  # type: ignore
        return graph

    @classmethod
    def create_empty(cls) -> "Measurement":
        """Create an empty measurement

        Returns:
            Measurement: empty measurement
        """
        return Measurement(channels=[], step_items=[], relations=[], log_channels=[])


def step_arrays(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Expand step value dictionary arrays to go through all needed combinations of values

    i.e.

    arrays = {"X": np.array([1, 2, 3]), "Y": np.array([4, 5, 6])}
    step_arrays = step_arrays(arrays)
    step_arrays = {
        "X": np.array([1, 2, 3, 1, 2, 3, 1, 2, 3]),
        "Y": np.array([4, 4, 4, 5, 5, 5, 6, 6, 6])
    }

    Args:
        arrays (dict[str, np.ndarray]):
            dictionary of step item values to expand into all needed value combinations

    Returns:
        dict[str, np.ndarray]:
            dictionary of step item values expanded into all needed value combinations
    """
    arrays = dict(reversed(arrays.items()))
    for array in arrays.values():
        assert array.ndim == 1
    array_shape = [array.shape[0] for array in arrays.values()]
    new_arrays = {
        name: (
            np.ones(array_shape)
            * array[
                tuple(
                    slice(None) if i == j else None for i, _ in enumerate(array_shape)
                )
            ]
        ).reshape(-1)
        for j, (name, array) in enumerate(arrays.items())
    }
    return dict(reversed(new_arrays.items()))
