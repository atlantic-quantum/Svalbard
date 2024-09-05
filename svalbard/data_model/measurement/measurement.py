"""
Data model for a measurement.
A measurement model is a collection of
  - channels used in a measurements.
  - step items that indicate how to step a subset of the channels,
  - relations that indicate how to set a different subset of the channels based on
    the values of the step items.
  - log channels that indicate which channels to log during the measurement.

"""

from typing import Any, Self

import numpy as np
import rustworkx as rx
from deprecated import deprecated
from pydantic import BaseModel, field_validator, model_validator

from ...typing import TSettingValueSweep
from .channel import Channel
from .log_channel import LogChannel
from .relation import RelationSettings
from .step_config import StepConfig, StepUnits
from .step_item import StepItem
from .step_range import RangeTypes, StepRange


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

    # pylint: disable=too-many-public-methods

    channels: list[Channel]
    step_items: list[StepItem]
    relations: list[RelationSettings]
    log_channels: list[LogChannel]

    @model_validator(mode="before")  # type: ignore
    @classmethod
    def validate_log_channels(cls, data: Any):
        """Convert log channel names to LogChannel objects"""
        for i, log_channel in enumerate(data["log_channels"]):
            if isinstance(log_channel, str):
                data["log_channels"][i] = LogChannel(name=log_channel)
        return data

    @field_validator("*")
    @classmethod
    def validate_names_unique(
        cls,
        v: list[RelationSettings] | list[StepItem] | list[Channel] | list[LogChannel],
    ):
        """
        validate that all channels, relation names, log channels and step item names are
        unique
        """
        names = {relation.name for relation in v}
        assert len(names) == len(v)
        return v

    @model_validator(mode="after")
    def validate_names_exist(self) -> Self:
        """validate that all relation, log channels and step items exist as channels"""
        for lst in [self.relations, self.step_items, self.log_channels]:
            lst: list[RelationSettings] | list[StepItem] | list[LogChannel]
            for item in lst:
                if item.name not in self.channel_names:
                    err = f"{item.__class__.__name__} {item.name}"
                    err += " uses on non-existing channel"
                    raise ValueError(err)
        return self

    @field_validator("step_items")
    @classmethod
    def validate_step_items_indices(cls, step_items: list[StepItem]) -> list[StepItem]:
        """validate that all step items have unique indices"""
        indicies = [step.index for step in step_items if step.index is not None]
        max_index = max(indicies, default=-1)
        for step in step_items:
            if step.index is None:
                step.index = max_index + 1
                max_index += 1
        return step_items

    @field_validator("step_items")
    @classmethod
    def validate_step_items_indices_order(
        cls, step_items: list[StepItem]
    ) -> list[StepItem]:
        """
        validate that all hardware sweepable step items have index < non HW sweepable
        """
        min_non_hw_index = float("inf")
        max_hw_index = float("-inf")
        for step in step_items:
            if step.hw_swept:
                max_hw_index = max(max_hw_index, step.index)  # type: ignore
            else:
                min_non_hw_index = min(min_non_hw_index, step.index)  # type: ignore
        if max_hw_index >= min_non_hw_index:
            raise ValueError(
                "Hardware sweepable step items must have index < non hardware sweepable"
            )
        return step_items

    def add_step(
        self,
        channel: str | Channel,
        values: np.ndarray | TSettingValueSweep,
        index: int | None = None,
        unit: str = "",
        hw_swept: bool = False,
    ):
        """
        Adds a step item to the list of step items by specifying a channel name
        (or Channel object), a list of values and an index. A new step item is created
        and inserted at the given index in the list of step items.

        Args:
            channel (str | Channel):
                name of the channel (or Channel object) to add a step item for
            values (np.ndarray | TSettingValueSweep):
                values to step through in the step item
            index (int):
                index to insert the new step item at
                Defaults to None. If None the step item is appended to the end of list.
            unit (str):
                unit of values, defaults to empty string
            hw_swept (bool):
                whether the step item is sweepable by hardware, defaults to False
        Raises:
            ValueError:
                if the channel name does not exist in the list of channels
        """
        if isinstance(channel, str):
            channel = self.get_channel(channel)
        if channel.name not in self.channel_names:
            self.add_channel(channel)
        if isinstance(values, np.ndarray):
            values = values.tolist()
        if not isinstance(values, list):
            values = [values]
        assert isinstance(values, list)

        step_item = StepItem(
            name=channel.name,
            config=StepConfig(
                step_unit=StepUnits.PHYSICAL if unit else StepUnits.INSTRUMENT
            ),
            ranges=[StepRange(range_type=RangeTypes.VALUES, values=values)],
            index=index,
            hw_swept=hw_swept,
        )

        self.add_step_item(step_item)

    @property
    def relation_names(self) -> list[str]:
        """return list of relation names"""
        return [relation.name for relation in self.relations]

    @property
    def step_item_names(self) -> list[str]:
        """return list of step item names"""
        return [step.name for step in self.step_items]

    @property
    def channel_names(self) -> list[str]:
        """return list of channel names"""
        return [channel.name for channel in self.channels]

    @property
    def log_channels_names(self) -> list[str]:
        """return list of log channel names"""
        return [log_channel.name for log_channel in self.log_channels]

    @property
    def max_step_index(self) -> int:
        """return the maximum index of all step items"""
        indicies = [step.index for step in self.step_items if step.index is not None]
        return max(indicies, default=-1)

    @property
    @deprecated(
        version="0.1.0",
        reason="Different logs can now have different shapes, use log_shapes instead"
        "\nremove in version 0.2.0",
    )
    def log_shape(self) -> tuple[int, ...]:
        """Returns default the shape of the log data"""
        return self.swept_shape

    @property
    def swept_shape(self) -> tuple[int, ...]:
        """Returns the shape swept by the step items"""
        if not self.step_items:
            # Empty step items means nothing is stepped, still log one point
            return (1,)
        self.check_matching_sizes()
        idx_size = {stp.index: len(stp.calculate_values()) for stp in self.step_items}
        step_item_shape = tuple([size for _, size in sorted(idx_size.items())])
        return tuple(step_item_shape)

    def check_matching_sizes(self) -> bool:
        """
        Check if all step items with the same index have the same size

        raises:
            ValueError: if step items with the same index have different sizes
        """
        if not self.step_items:
            return True
        names = [step.name for step in self.step_items]
        sizes = np.array([len(step.calculate_values()) for step in self.step_items])
        inds = np.array([step.index for step in self.step_items])
        for index in set(inds):
            if not np.equal(sizes[inds == index], sizes[inds == index][0]).all():
                err = f"Step items with index {index} have different sizes:\n"
                for name, size in zip(names, sizes):
                    if inds[sizes == size][0] == index:
                        err += f"  {name}: {size}\n"
                raise ValueError(err)
        return True

    @property
    def log_shapes(self) -> dict[str, tuple[int, ...]]:
        """Return the shape of the log data for each log channel"""
        self.check_matching_sizes()
        index_size = {
            stp.index: len(stp.calculate_values())
            for stp in self.step_items
            if stp.index is not None
        }
        index_steps: dict[int, set[str]] = {}
        for step in self.step_items:
            if step.index is not None:
                index_steps.setdefault(step.index, set()).add(step.name)
        idx_size_steps = {i: (s, index_steps[i]) for i, s in index_size.items()}
        return {log.name: log.shape(idx_size_steps) for log in self.log_channels}

    def add_log(self, log: str | LogChannel):
        """
        Add a channel to the list of channels to log

        Args:
            log (str | LogChannel):
                name of the channel or LogChannel object to add

        Raises:
            ValueError:
                if no channel with the given name exists in the Channel list
        """
        if isinstance(log, str):
            log = Channel.make_pythonic(log)
            log = LogChannel(name=log)
        if log.name not in self.channel_names:
            raise ValueError(f"Channel {log.name} does not exist")
        self.log_channels.append(log)

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
        if index is not None:
            step_item.index = index
        elif step_item.index is None:
            step_item.index = self.max_step_index + 1
        self.step_items.append(step_item)
        self.validate_step_items_indices_order(self.step_items)  # type: ignore

    def add_channel(self, channel: Channel):
        """
        Add a channel to the list of channels

        Args:
            channel (Channel):
                Channel to add

        Raises:
            ValueError:
                if a channel with the same name already exists
                of if a channel for the same instrument and setting pair aready exits.
        """
        if channel.name in self.channel_names:
            raise ValueError(f"Channel with name {channel.name} already exists")
        for o_channel in self.channels:
            if (
                o_channel.instrument_identity == channel.instrument_identity
                and o_channel.instrument_setting_name == channel.instrument_setting_name
            ):
                raise ValueError(
                    f"Channel '{o_channel.name}' for instrument "
                    f"{channel.instrument_identity} and setting "
                    f"{channel.instrument_setting_name} already exits"
                )
        self.channels.append(channel)

    def get_channel(
        self, name: str | StepItem | RelationSettings | LogChannel | Channel
    ) -> Channel:
        """
        returns a channel object by name or by a step item, relation, log channel or
        channel object

        Args:
            name (str| StepItem | RelationSettings | LogChannel | Channel):
                name of the channel to get or a step item, relation, log channel or
                channel object

        Raises:
            ValueError:
                if no channel with the given name exists in the Channel list

        Returns:
            Channel: channel object with the given name
        """
        if not isinstance(name, str):
            name = name.name
        name = Channel.make_pythonic(name)
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
        name = Channel.make_pythonic(name)
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
        name = Channel.make_pythonic(name)
        for relation in self.relations:
            if relation.name == name:
                return relation
        raise ValueError(f"Relation {name} does not exist")

    def get_log_channel(self, name: str) -> LogChannel:
        """
        Return a LogChannel object by name

        Args:
            name (str):
                name of the log channel to get

        Raises:
            ValueError:
                if no log channel with the given name exists in the log channel list

        Returns:
            LogChannel: log channel object with the given name
        """
        name = Channel.make_pythonic(name)
        for log_channel in self.log_channels:
            if log_channel.name == name:
                return log_channel
        raise ValueError(f"Log channel {name} does not exist")

    def remove_channel(self, name: str):
        """
        Remove a channel from the list of channels by name

        Args:
            name (str):
                name of the channel to remove
        """
        name = Channel.make_pythonic(name)
        channel = self.get_channel(name)
        self.channels.remove(channel)

    def remove_step_item(self, name: str):
        """
        Remove a step item from the list of step items by name

        Args:
            name (str):
                name of the step item to remove
        """
        name = Channel.make_pythonic(name)
        step_item = self.get_step_item(name)
        self.step_items.remove(step_item)

    def remove_relation(self, name: str):
        """
        Remove a relation from the list of relations by name

        Args:
            name (str):
                name of the relation to remove
        """
        name = Channel.make_pythonic(name)
        relation = self.get_relation(name)
        self.relations.remove(relation)

    def remove_log(self, name: str):
        """
        Remove a log channel from the list of log channels by name

        Args:
            name (str):
                name of the log channel to remove
        """
        name = Channel.make_pythonic(name)
        log = self.get_log_channel(name)
        self.log_channels.remove(log)

    def set_log_position(self, name: str, index: int):
        """
        Change the position of a log channel in the list of log channels

        Args:
            name (str):
                name of the log channel to move
            index (int):
                new index of the log channel
        """
        name = Channel.make_pythonic(name)
        log = self.get_log_channel(name)
        self.log_channels.remove(log)
        self.log_channels.insert(index, log)

    def set_step_item_position(self, name: str, index: int):
        """
        Change the position of a step item in the list of step items

        Args:
            name (str):
                name of the step item to move
            index (int):
                new index of the step item
        """
        name = Channel.make_pythonic(name)
        step_item = self.get_step_item(name)
        self.step_items.remove(step_item)
        self.step_items.insert(index, step_item)
        self.validate_step_items_indices_order(self.step_items)  # type: ignore

    def set_step_item_index(self, name: str, index: int):
        """
        Change the index of a step item in the list of step items

        Args:
            name (str):
                name of the step item to move
            index (int):
                new index of the step item
        """
        name = Channel.make_pythonic(name)
        step_item = self.get_step_item(name)
        step_item.index = index

    def _calculate_relation_values(
        self, step_values: dict[str, np.ndarray], expanded_arrays: bool = True
    ) -> dict[str, np.ndarray]:
        """
        Calculate values for all relations in the measurement.

        Args:
            step_values (dict[str, np.ndarray]):
                dictionary of all step item values with the channel name as key and the
                values as numpy array.
            expanded_arrays (bool):
                if True the step item arrays have been expanded to go through all needed
                combinations of values, if False each step item value array is just the
                values that step item should step through in order


        Returns:
            dict[str, np.ndarray]:
                dictionary of all relation values with the channel name as key and the
                values as numpy array.
        """
        if not self.relations:
            return {}

        # if the step arrays are expanded the relation dependencies are the step values
        relation_dependencies = step_values
        step_indicies = [s.index for s in self.step_items if s.index is not None]
        step_indicies = sorted(set(step_indicies))
        if not expanded_arrays:
            # if the step arrays are not expanded the indices of the step items
            # need to be considered to calculate the relation values correctly
            relation_dependencies = {}
            for step in self.step_items:
                assert step.index is not None
                dependency_shape = np.ones(len(self.swept_shape), dtype=int)
                dependency_shape[step_indicies.index(step.index)] = len(
                    step_values[step.name]
                )
                relation_dependencies[step.name] = step_values[step.name].reshape(
                    dependency_shape
                )

        dependency_graph = self.create_relations_dependency_graph()
        rel_dict = {relation.name: relation for relation in self.relations}
        relation_values: dict[str, np.ndarray] = {}
        for node_index in rx.topological_sort(dependency_graph):
            name = dependency_graph.nodes()[node_index]
            if name in rel_dict:
                if rel_dict[name].enable:
                    relation_values[name] = rel_dict[name].calculate_values(
                        relation_dependencies
                    )
        return relation_values

    @staticmethod
    def _swap_lists_for_index_arrays(
        step_values: dict[str, np.ndarray | TSettingValueSweep]
    ) -> tuple[dict[str, np.ndarray], dict[str, TSettingValueSweep]]:
        """Replace list values with index arrays

        Args:
            step_values (dict[str, np.ndarray |  TSettingValueSweep]):
                dictionary of step item values with the channel name as key and the
                values as numpy array or a list of values.

        Returns:
            tuple[dict[str, np.ndarray], dict[str, TSettingValueSweep]]:
                Dictionaries of
                    a) step item values with the channel name as key and the values as
                       numpy array, values that were lists in the input dictionary are
                       replaced with index arrays.
                    b) list values that were replaced with index arrays, with the
                       channel name as key and the values as a list of values.
        """
        index_values: dict[str, TSettingValueSweep] = {}
        _step_values: dict[str, np.ndarray] = {}
        for name, values in step_values.items():
            if isinstance(values, np.ndarray):
                _step_values[name] = values
                continue
            index_values[name] = values
            _step_values[name] = np.arange(len(values))
        return _step_values, index_values

    @staticmethod
    def _swap_index_arrays_for_lists(
        step_values: dict[str, np.ndarray],
        list_values: dict[str, TSettingValueSweep],
    ) -> dict[str, np.ndarray | TSettingValueSweep]:
        """
        Replace index arrays with list values, if the values were expanded the resulting
        lists will be in the same order as the expanded arrays.

        Args:
            step_values (dict[str, np.ndarray]):
                dictionary of step item values with the channel name as key and the
                values as numpy array. some values may be index arrays.
            list_values (dict[str, TSettingValueSweep]):
                dictionary of list values that were replaced with index arrays, with the
                channel name as key and the values as a list of values.

        Returns:
            dict[str, np.ndarray | TSettingValueSweep]:
                dictionary of step item values with the channel name as key and the
                values as numpy array or a list of values. A list is returned if the
                step item values are lists of uneven lengths (i.e. the list of lists
                can't be converted to a numpy array). The values that were index arrays
                are replaced with the original list values.
        """
        _step_values: dict[str, np.ndarray | TSettingValueSweep] = {}
        for name in step_values:
            if name not in list_values:
                _step_values[name] = step_values[name]
                continue
            values = list_values[name]
            try:
                values = np.array(values)
                _step_values[name] = values[step_values[name]]
            except ValueError:
                list_step_values: TSettingValueSweep = [
                    values[index] for index in step_values[name]
                ]
                _step_values[name] = list_step_values
        return _step_values

    def calculate_values(
        self, expand_arrays: bool = True
    ) -> dict[str, np.ndarray | TSettingValueSweep]:
        """
        Calculate values for all step items and relations in the measurement.

        Returns:
            dict[str, np.ndarray | TSettingValueSweep]]:
                dictionary of all step item and relation values
                with the channel name as key and the values as numpy array or a list
                of values. A list is returned if the step item values are
                lists of uneven lengths (i.e. the list of lists can't be converted to
                a numpy array).
            expand_arrays (bool):
                if True expand step item arrays to go through all needed combinations
                of values, if False return the step item values as is.
        """

        self.check_matching_sizes()
        step_values = {step.name: step.calculate_values() for step in self.step_items}
        # the step arrays function requires 1D arrays so list are replaced with indices
        step_values, list_values = self._swap_lists_for_index_arrays(step_values)
        idx = [step.index for step in self.step_items if step.index is not None]
        step_values = step_arrays(step_values, idx) if expand_arrays else step_values

        # Relations can't depend on step items with list values
        relation_values = self._calculate_relation_values(step_values, expand_arrays)
        step_values.update(relation_values)

        # swap back the index arrays for lists
        step_values = self._swap_index_arrays_for_lists(step_values, list_values)
        return step_values

    def calculate_dataset_values(self) -> dict[str, np.ndarray]:
        """
        Calculate axes values for all step items and relations in the measurement.
        These are the values that are used when creating datasets for the measurement
        step items and relations.

        Step items that have list values that can't be converted to numpy arrays (i.e.
        values of step item are lists of uneven lengths.) are are converted to index
        arrays.

        Returns:
            dict[str, np.ndarray]:
                dictionary of all step item and relation axes values, with the channel
                name as key and the values as numpy array.

        """
        step_values = self.calculate_values(False)
        for name, values in step_values.items():
            if not isinstance(values, list):
                continue
            step_item = self.get_step_item(name)
            step_values[name] = np.arange(step_item.step_count)
        return step_values  # type: ignore

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
    def create_empty(cls) -> Self:
        """Create an empty measurement

        Returns:
            Measurement: empty measurement
        """
        return cls(channels=[], step_items=[], relations=[], log_channels=[])


def step_arrays(
    arrays: dict[str, np.ndarray], indicies: list[int] | None = None
) -> dict[str, np.ndarray]:
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
    values = meshgrid(*arrays.values(), indicies=indicies)
    return {name: value.T.flatten() for name, value in zip(arrays.keys(), values)}


def meshgrid(*xi, indicies: list[int] | None = None) -> list[np.ndarray]:
    """
    Specialized version of numpy.meshgrid that takes dimension indicies into account
    i.e. can have multiple 1-D vectors for the same dimension.

    Args:
        (*xi) x1, x2,..., xn (array like):
            1-D arrays representing the coordinates of a grid.
        indicies (list[int] | None, optional):
            The dimension index of each of the 1-D arrays. Defaults to None.
            If None the dimension index is assumed to be the same as the order of the
            1-D arrays.

    Raises:
        ValueError:
            if the length of indicies is not the same as the length of 1-D arrays.

    Returns:
        list[np.ndarray]: Meshgrid of the input arrays expanded to N-D arrays

        Example:

        xi = [np.array([1, 2, 3]), np.array([4, 5]), np.array([6, 7, 8])]
        indicies = [0, 1, 0]

        arrs = meshgrid(xi, indicies)

        arrs[0].T
        -> array([[1, 2, 3],
                  [1, 2, 3]]),

        arrs[1].T
        -> array([[4, 4, 4],
                  [5, 5, 5]]),

        arrs[2].T
        -> array([[6, 7, 8],
                  [6, 7, 8]]),

    """

    indicies = indicies or list(range(len(xi)))
    if len(indicies) != len(xi):
        raise ValueError("indicies must have the same length as xi")
    ndim = len(indicies)

    # Determine the uniqe dimension indicies
    u_indicies = sorted(set(indicies))
    # The the dimension order of each 1-D array / index
    index_order = [u_indicies.index(i) for i in indicies]
    s0 = (1,) * ndim

    # Turn the shape of each 1-D array (n,) into (1, ..., -1, ..., 1), the -1 is where
    # the dimension index is, the rest of the dimensions are 1
    output = [
        np.asanyarray(x).reshape(s0[:i] + (-1,) + s0[i + 1 :])
        for i, x in zip(index_order, xi)
    ]

    # By broadcasting the 1-D vectors together we get the full N-D matrix
    return np.broadcast_arrays(*output, subok=True)
