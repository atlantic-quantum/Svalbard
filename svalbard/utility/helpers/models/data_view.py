from enum import Enum
from pathlib import Path
from typing import TypeVar

import numpy as np
from pydantic import BaseModel, PrivateAttr, ValidationError

from ...data_model.measurement.channel import Channel
from ...data_model.measurement.log_channel import LogChannel
from ...data_model.measurement.measurement import Measurement
from ...data_model.measurement.step_item import StepConfig, StepItem, StepRange
from ...utility import str_helper
from .plotting.models import Colormap, ImageGraphModel


class DataOperations(Enum):
    """Enum for data operations"""

    NONE = "None"
    SUBTRACT_MEAN = "y-<y>"
    NORMALIZE = "Normalize"
    DYDX = "d/dx"
    FFT = "FFT"
    HISTOGRAM = "Histogram"
    HISTOGRAM_IQ = "IQ Histogram"


class ComplexView(Enum):
    """Enum for complex view"""

    REAL = "Real"
    IMAG = "Imag"
    MAG = "Magnitude"
    PHASE = "Phase"


def calc_rotation_angle(data):
    """Calculate angle that will rotate complex data to the real component"""
    # data conditioning
    data = data[np.isfinite(data)]
    if len(data) < 2:
        return 0.0
    if not np.any(np.iscomplex(data)):
        return 0.0
    poly_1 = np.polyfit(data.real, data.imag, 1)
    poly_2 = np.polyfit(data.imag, data.real, 1)
    # get slope either as dy/dx and dx/dy
    if abs(poly_1[0]) < abs(poly_2[0]):
        angle = np.arctan(poly_1[0])
    else:
        angle = np.pi / 2.0 - np.arctan(poly_2[0])
        if angle > np.pi:
            angle -= np.pi
    angle = -angle
    # make peaks instead of dips
    data = np.real(data * np.exp(1j * angle))
    mean_value = np.mean(data)
    # get final angle
    first = abs(data[0] - mean_value)
    low = abs(np.min(data) - mean_value)
    high = abs(np.max(data) - mean_value)
    if first > 0.5 * max(low, high):
        if data[0] < mean_value:
            angle += np.pi
    else:
        if high < low:
            angle += np.pi
    return angle


def calculate_phase_delay(x: np.ndarray, y: np.ndarray):
    """Calculate phase delay for (x,y) complex data.

    Args:
        x (np.ndarray): x data, typically frequency
        y (np.ndarray): complex y data, typically S21

    Returns:
        float: phase delay in radians/x units
    """
    # get unwrapped phase
    phase = np.unwrap(np.angle(y))
    # fit to 1D polynomial, return negative slope
    linear_fit = np.polyfit(x, phase, 1)
    return -linear_fit[0]


def smooth(x, n=3):
    """Smooth data with moving average"""
    if n < 2:
        return x
    # number of points to average can't be longer than data
    n = min(n, len(x))
    # pad data to avoid issues at the edges
    s = np.r_[2 * x[0] - x[n - 1 :: -1], x, 2 * x[-1] - x[-1:-n:-1]]
    w = np.ones(n)
    y = np.convolve(s, w / np.nansum(w), mode="same")
    return y[n : -n + 1]


def add_step_items_for_log_channels(measurement: Measurement):
    """Add step items for log channels with non-scalar data

    Note that the function updates the measurement object in place.
    """
    # get index of lowest step item, new ones will be added before this
    step_items = sorted(measurement.step_items, key=lambda x: x.index)
    indicies = {s.index for s in step_items if s.index is not None}
    data_index = min(indicies) - 1 if len(indicies) > 0 else 0

    # keep track of new step item names in a dict, key is step item name
    new_names: dict[str, LogChannel] = {}
    # add step items for log channels with non-scalar data
    for log_channel in measurement.log_channels:
        # ignore scalar data
        if log_channel.base_shape == (1,):
            continue
        # add step item for each dimension of data
        for n, size in enumerate(log_channel.base_shape):
            if len(log_channel.base_shape) > 1:
                name = log_channel.name + f"_dim{n}"
            else:
                name = log_channel.name + "_index"
            # add extra channel if not already added
            if name in measurement.channel_names:
                continue
            # add channel for new step item
            log_channel_obj = measurement.get_channel(log_channel)
            extra_step_channel = Channel(
                name=name,
                unit_physical="",
                instrument_setting_name=name,
                instrument_identity=log_channel_obj.instrument_identity,
            )
            measurement.add_channel(extra_step_channel)
            # add extra step item
            step_range = StepRange(start=0, stop=size - 1, step_count=size)
            step_item = StepItem(
                name=name,
                config=StepConfig(),
                ranges=[step_range],
                hw_swept=True,
            )
            measurement.add_step_item(step_item, data_index)
            data_index -= 1
            # store new step item name
            new_names[name] = log_channel

    # after adding step items, old log channels need to exclude new step items
    for log_channel in measurement.log_channels:
        # fix issue with inclusive log channels with no step items, make exclusive
        if log_channel.inclusive and not log_channel.step_names:
            log_channel.inclusive = False
        # if not inclusive, add new step items to exclude list
        if not log_channel.inclusive:
            log_channel.step_names.update(set(new_names.keys()))
    # finally, include new step items in relevant log channels
    for step_name, log_channel in new_names.items():
        if log_channel.inclusive:
            log_channel.step_names.add(step_name)
        else:
            log_channel.step_names.discard(step_name)


def get_primary_step_items(measurement: Measurement) -> list[StepItem]:
    """
    Returns list of StepItems that defines the data dimensions of a measurement.

    If multiple StepItems have the same index, only the first one is returned.

    Returns:
        list[StepItem]: List of StepItems defining the data dimensions
    """
    # get list of all primary step items and set with all indicies
    step_items = sorted(measurement.step_items, key=lambda x: x.index)
    indicies = {s.index for s in step_items if s.index is not None}
    primary_step_items = []
    for step in step_items:
        # only add the first step item with a given index
        if step.index not in indicies:
            continue
        primary_step_items.append(step)
        indicies.remove(step.index)
    return primary_step_items


ndarray = TypeVar("ndarray")


class GraphData(BaseModel):
    """Model for graph data, allowing a LogView object to define all graph contents."""

    # model_config = ConfigDict(arbitrary_types_allowed=True)
    # define data arrays
    x: ndarray = np.array([])
    y: ndarray = np.array([])
    y_complex: ndarray = np.array([])  # complex data before picking component
    z: ndarray = np.array([])
    # define labels
    x_name: str = ""
    x_unit: str = ""
    y_name: str = ""
    y_unit: str = ""
    z_name: str = ""
    z_unit: str = ""
    # define preferred plot to show.  If None, let user choose in UI
    image_plot: bool = None


class XYSelector(BaseModel):
    """Pydantic model for XY selection, for use in the Viewer app."""

    # keep track of selection as string
    x: str = ""
    y: str = ""
    z: str = ""
    # basic data analysis parameters
    complex_view: ComplexView = ComplexView.MAG
    auto_rotate: bool = False
    rotation: float = 0.0
    auto_delay: bool = False
    delay: float = 0.0
    unwrap: bool = False
    plot_db: bool = False
    operation: DataOperations = DataOperations.NONE
    bins: int = 51
    smooth: int = 1
    # keep track of selected data type from last call
    _is_complex: bool = False

    def is_data_complex(self) -> bool:
        """Return True if data used in last call is complex"""
        return self._is_complex

    def apply_complex_view(self, data: GraphData):
        """Apply complex view to data"""
        # store if data is complex, will be used to update widget
        self._is_complex = data.y.dtype == np.complex128
        # return immediately if data is not complex
        if not self._is_complex:
            return
        # phase rotation
        if self.auto_rotate:
            for n in range(data.y.shape[0]):
                theta = calc_rotation_angle(data.y[n, :])
                data.y[n, :] = data.y[n, :] * np.exp(1j * theta)
            # update rotation parameter in UI
            self.rotation = theta
        else:
            # apply fixed rotation
            data.y = data.y * np.exp(1j * self.rotation)
        # phase delay
        if self.auto_delay:
            for n in range(data.y.shape[0]):
                xx = data.x[n, :] if len(data.x.shape) == 2 else data.x
                delay = calculate_phase_delay(xx, data.y[n, :])
                data.y[n, :] = data.y[n, :] * np.exp(1j * delay * xx)
            # update delay parameter in UI
            self.delay = delay
        else:
            # apply fixed delay
            for n in range(data.y.shape[0]):
                xx = data.x[n, :] if len(data.x.shape) == 2 else data.x
                data.y[n, :] = data.y[n, :] * np.exp(1j * self.delay * xx)
        # store complex data in separate variable, for use in histogram
        data.y_complex = data.y
        # pick complex component to view
        if self.complex_view == ComplexView.REAL:
            data.y_name += "- Real"
            data.y = data.y.real
        elif self.complex_view == ComplexView.IMAG:
            data.y_name += "- Imag"
            data.y = data.y.imag
        elif self.complex_view == ComplexView.MAG:
            data.y_name += "- Magnitude"
            data.y = np.abs(data.y)
        elif self.complex_view == ComplexView.PHASE:
            data.y = np.angle(data.y)
            data.y_name += "- Phase"
            data.y_unit = "rad"
            if self.unwrap:
                data.y = np.unwrap(data.y)
        # convert to dB
        if self.plot_db:
            data.y = 20 * np.log10(data.y)
            data.y_unit = "dB"

    def apply_operations(self, data: GraphData):
        """Apply operations to data"""
        # go through operations, step-by-step
        if self.operation == DataOperations.SUBTRACT_MEAN:
            data.y -= np.nanmean(data.y, axis=1, keepdims=True)
        elif self.operation == DataOperations.NORMALIZE:
            for n in range(data.y.shape[0]):
                ymax = np.nanmax(data.y[n, :])
                ymin = np.nanmin(data.y[n, :])
                if ymax != ymin:
                    data.y[n, :] = (data.y[n, :] - ymin) / (ymax - ymin)
        elif self.operation == DataOperations.DYDX:
            for n in range(data.y.shape[0]):
                dx = np.gradient(data.x[n, :] if len(data.x.shape) == 2 else data.x)
                # make sure dx is not zero
                zero_dx = np.nonzero(dx == 0)[0]
                dx[zero_dx] = 1
                data.y[n, :] = np.gradient(data.y[n, :]) / dx
                data.y[n, :][zero_dx] = 0.0
            data.y_name = f"d({data.y_name})/d({data.x_name})"
            data.y_unit = f"{data.y_unit}/{data.x_unit}"
        elif self.operation == DataOperations.FFT:
            y_out = []
            x_out = []
            for n in range(data.y.shape[0]):
                xx = data.x[n, :] if len(data.x.shape) == 2 else data.x
                x_fft = np.fft.fftfreq(len(data.y[n, :]), xx[1] - xx[0])
                y_fft = np.fft.rfft(data.y[n, :])
                # ignore DC and negative frequencies
                x_fft = np.abs(x_fft[1 : len(y_fft)])
                y_fft = np.abs(y_fft[1:])
                # normalize
                if len(y_fft) > 0:
                    y_fft = y_fft / len(y_fft)
                y_out.append(y_fft)
                x_out.append(x_fft)
            data.y = np.array(y_out)
            data.x = np.array(x_out)
            data.y_name = f"{data.y_name} - FFT"
            data.x_name = f"{data.x_name} - FFT"
            data.x_unit = "Hz"
        elif self.operation == DataOperations.HISTOGRAM:
            y_out = []
            x_out = []
            y_min = np.nanmin(data.y)
            y_max = np.nanmax(data.y)
            for n in range(data.y.shape[0]):
                (y_hist, bin_edges) = np.histogram(
                    data.y[n, :], bins=self.bins, range=(y_min, y_max)
                )
                # calculate bin centers
                x_hist = (bin_edges[0:-1] + bin_edges[1:]) / 2
                # convert array to float
                y_out.append(np.array(y_hist, dtype=float))
                x_out.append(x_hist)
            data.y = np.array(y_out)
            data.x = np.array(x_out)
            data.x_name = data.y_name
            data.x_unit = data.y_unit
            data.y_name = "Count"
            data.y_unit = ""
        elif self.operation == DataOperations.HISTOGRAM_IQ:
            y_all = data.y_complex.flatten()
            (hist_data, data.x, data.z) = np.histogram2d(
                y_all.real,
                y_all.imag,
                bins=self.bins,
                range=(
                    [np.nanmin(y_all.real), np.nanmax(y_all.real)],
                    [np.nanmin(y_all.imag), np.nanmax(y_all.imag)],
                ),
            )
            # if plotting in dB, add 0.5 to avoid log(0)
            if self.plot_db:
                hist_data = 10.0 * np.log10(hist_data + 0.5)
            data.y = hist_data.T
            # set labels
            data.x_name = f"{data.y_name} - I"
            data.x_unit = data.y_unit
            data.z_name = f"{data.y_name} - Q"
            data.z_unit = data.y_unit
            data.y_name = "Count"
            data.y_unit = ""
            # make sure image plot is used
            data.image_plot = True

        # smooth data
        if (
            self.operation
            not in (DataOperations.HISTOGRAM, DataOperations.HISTOGRAM_IQ)
            and self.smooth > 1
        ):
            for n in range(data.y.shape[0]):
                data.y[n, :] = smooth(data.y[n, :], self.smooth)


class LogView(BaseModel):
    """Pydantic model for views of measurement data, for use in the Viewer app."""

    xy_selector: XYSelector = XYSelector()
    # entry list/column selection
    use_entry_list: bool = True
    selected_rows: list[int] = []
    selected_cols: dict[str, list[int]] = {}
    # general view settings
    image: bool = False
    # graph settings
    graph: ImageGraphModel = ImageGraphModel()
    # a view is connected to a single measurement
    _measurement: Measurement = PrivateAttr()
    # cache outer step values for faster access
    _map_primary_to_all: dict[str, str] = PrivateAttr()
    _map_all_to_primary: dict[str, str] = PrivateAttr()
    _outer_step_values: dict[str, np.ndarray] = PrivateAttr()
    _labels: dict[str, list[str]] = None

    def initialize_view(self, measurement: Measurement, set_defaults: bool = False):
        """Initialize view and update values to be compatible with given measurement

        Parameters
        ----------
        measurement : Measurement
            Measurement object to be viewed
        set_defaults : bool, optional
            Override all values with default view, by default False
        """
        # add view-based step items and store measurement object
        add_step_items_for_log_channels(measurement)
        self._measurement = measurement
        # cache mappings between primary and all other step items
        self._map_primary_to_all = {}
        self._map_all_to_primary = {}
        for primary_step in get_primary_step_items(measurement):
            # primary step item is mapping to all step items with same index
            self._map_primary_to_all[primary_step.name] = [
                s.name for s in measurement.step_items if s.index == primary_step.index
            ]
            # the seconday step items should map to the primary step item
            for s in self._map_primary_to_all[primary_step.name]:
                self._map_all_to_primary[s] = primary_step.name
        # set default view settings based on measurement, if wanted
        if set_defaults:
            # default view is first log channel vs first step item
            self.xy_selector.y = self.get_default_y_selection()
            self.xy_selector.x = self.get_default_x_selection()
            self.xy_selector.z = self.get_default_z_selection()
            # show as image if more than 5 traces
            shape = measurement.log_shapes[self.xy_selector.y]
            self.image = bool(len(shape) >= 2 and shape[1] > 5)
            # use column view for higher-dimensional data
            if len(shape) > 2:
                self.use_entry_list = False
            # set default entry selection
            self.select_default_entries()
        # set default colormap for image if none given
        if self.graph.colormap is None:
            self.graph.colormap = Colormap.RdBu
        # validate xy selector, will update cache with outer step values
        self.validate_xy_selector()

    def select_default_entries(self):
        """Set default entry selection for view"""
        # show as image if more than 5 traces
        if self.image:
            shape = self._measurement.log_shapes[self.xy_selector.y]
            self.selected_rows = list(range(shape[1]))
        else:
            self.selected_rows = [0]
        # clear column selection
        self.selected_cols = {}

    def validate_xy_selector(self):
        """Validate xy selector settings, update to defaults if needed"""
        # check if x selection is valid
        allowed_x_names = self.get_x_channel_names()
        # reset x-selector if selection not compatible with y-selector
        if self.xy_selector.x not in allowed_x_names:
            self.xy_selector.x = self.get_default_x_selection()
            # clear entry selection to default if x selection changes
            self.select_default_entries()
        # same for z selection
        allowed_z_names = self.get_z_channel_names()
        if len(allowed_z_names) > 0 and self.xy_selector.z not in allowed_z_names:
            self.xy_selector.z = self.get_default_z_selection()
        # check that x selection is consistent with entry vs column view
        if self.use_entry_list:
            # get primary steps for current x selection
            primary_x_name = self._map_all_to_primary[self.xy_selector.x]
            allowed_steps = self.get_active_step_items()
            # if x-selection is not first primary step, switch to column view mode
            if primary_x_name != allowed_steps[0].name:
                self.use_entry_list = False
        # after validation, update cached values
        self.cache_outer_step_values()

    def cache_outer_step_values(self):
        """Cache step values for faster access, run after changing x or y selector"""
        # cache outer step indices for faster access
        step_items = self.get_active_step_items(include_x=False)
        # use meshgrid to expand indices to all entries
        ranges = [np.arange(len(step.calculate_values())) for step in step_items]
        step_indices = [v.T.flatten() for v in np.meshgrid(*ranges, indexing="ij")]
        # using step indices, generate values for outer all steps
        self._outer_step_values = {}
        for n, primary_step in enumerate(step_items):
            for step_name in self._map_primary_to_all[primary_step.name]:
                step = self._measurement.get_step_item(step_name)
                values = step.calculate_values()
                if isinstance(values, np.ndarray):
                    outer_values = values[step_indices[n]]
                else:
                    # if values are not numpy arrays, step items are list
                    # for plotting purposes, just create a list of incremental values
                    values = np.arange(len(values))
                    outer_values = values[step_indices[n]]
                self._outer_step_values[step.name] = outer_values
        # clear label cache
        self._labels = None

    def get_active_step_items(
        self, include_x=True, include_z=True, include_secondary=False
    ) -> list[StepItem]:
        """Returns list of StepItems that defines the data dimensions of current view

        Parameters
        ----------
        include_x : bool, optional
            Include step item defining x-channel in list, by default True
        include_z : bool, optional
            If image mode, include step defining z-channel in list, by default True
        include_secondary : bool, optional
            Whether to include secondary step items in the list, by default False

        Returns:
            list[StepItem]: List of active StepItems
        """
        primary_step_items = get_primary_step_items(self._measurement)
        # remove step items not relevant for current selection
        log_channel = self._measurement.get_log_channel(self.xy_selector.y)
        primary_include_exclude = [
            self._map_all_to_primary[name]
            for name in log_channel.step_names
            if name in self._map_all_to_primary
        ]
        if log_channel.inclusive:
            # inclusive mode, match included log channel list to primary step items
            primary_step_items = [
                step
                for step in primary_step_items
                if step.name in primary_include_exclude
            ]
        else:
            # exclusive mode, remove excluded log channel list from primary step items
            primary_step_items = [
                step
                for step in primary_step_items
                if step.name not in primary_include_exclude
            ]
        # get shape of relevant log channel
        log_channel = self._measurement.get_log_channel(self.xy_selector.y)
        # exclude step corresponding to x-selection, if requested
        if not include_x:
            step_item_x = self._measurement.get_step_item(
                self._map_all_to_primary[self.xy_selector.x]
            )
            primary_step_items.remove(step_item_x)
        # exclude step corresponding to z-selection, only possible in image mode
        if self.image and not include_z:
            step_item_z = self._measurement.get_step_item(
                self._map_all_to_primary[self.xy_selector.z]
            )
            primary_step_items.remove(step_item_z)
        # add secondary step items, if wanted
        if include_secondary:
            all_step_item_names = []
            for primary_step in primary_step_items:
                all_step_item_names.extend(self._map_primary_to_all[primary_step.name])
            primary_step_items = [
                self._measurement.get_step_item(name) for name in all_step_item_names
            ]
        return primary_step_items

    def save_view(self, file_path):
        """Save current view to default location on disk"""
        json_str = self.model_dump_json(exclude={"measurement"})
        with open(file_path, "w") as f:
            f.write(json_str)

    def get_default_y_selection(self) -> str:
        """Get default y-selection for data

        Returns:
            str: Default y-selection
        """
        return self._measurement.log_channels_names[-1]

    def get_default_x_selection(self) -> str:
        """Get default x-selection for data

        Returns:
            str: Default x-selection
        """
        return self.get_x_channel_names()[0]

    def get_default_z_selection(self) -> str:
        """Get default z-selection for data

        Returns:
            str: Default z-selection
        """
        z_channel_names = self.get_z_channel_names()
        if len(z_channel_names) > 0:
            return self.get_z_channel_names()[0]
        return ""

    def get_y_channel_names(self) -> list[str]:
        """Get list of data channels for populating dropdowns in UI

        Returns:
            list[str]: list of channel names
        """
        return self._measurement.log_channels_names

    def get_x_channel_names(self) -> list[str]:
        """Get list of data channels for populating dropdowns in UI

        Returns:
            list[str]: list of channel names allowed for given y-selection
        """
        # include all step items valid for this view
        step_items = self.get_active_step_items(include_secondary=True)
        # return step names
        all_step_item_names = [step.name for step in step_items]
        return all_step_item_names

    def get_z_channel_names(self) -> list[str]:
        """Get list of data channels for populating dropdowns in UI

        Returns:
            list[str]: list of channel names for given x- and y-selection
        """
        # include all primatry step items, except one used for x
        step_items = self.get_active_step_items(include_x=False, include_secondary=True)
        # return step names
        all_step_item_names = [step.name for step in step_items]
        return all_step_item_names

    def get_number_of_entries(self) -> int:
        """Get number of entries of data for current x-axis selection

        Returns:
            int: number of entries
        """
        size = 1
        for step in self.get_active_step_items(include_x=False):
            size *= len(step.calculate_values())
        return size

    def get_outer_step_values(self, name) -> np.ndarray:
        """Get outer step values for named channel for current view

        Args:
            name (str): name of channel

        Returns:
            dict[str, np.ndarray]: dictionary of outer step values
        """
        return self._outer_step_values[name]

    def get_label(self, entry: int) -> str:
        """Get descriptive label for given entry

        Args:
            entry (int): entry number

        Returns:
            str: label for entry
        """
        # limit labels to show to avoid slowdown, TODO generate labels on demand
        MAX_LABELS = 200
        # check if labels have been calculated
        if self._labels is None:
            # calculate labels
            self._labels = {}
            for step in self.get_active_step_items(include_x=False):
                values = self.get_outer_step_values(step.name)
                ch = self._measurement.get_channel(step.name)
                labels = []
                for value in values[:MAX_LABELS]:
                    if isinstance(value, list) or isinstance(value, np.ndarray):
                        labels.append(str(value))
                    else:
                        labels.append(
                            str_helper.get_si_string(
                                value, ch.unit_physical, decimals=6
                            )
                        )
                self._labels[step.name] = labels
        # return entry number if no labels are available
        if entry >= MAX_LABELS:
            return f"#{1 + entry}"
        # combine labels, starting with entry number
        labels = [f"{name}: {values[entry]}" for name, values in self._labels.items()]
        return ", ".join([f"#{1 + entry}"] + labels)

    def get_inner_dimension(self) -> int:
        """Get inner dimension of data for current view

        Returns:
            int: inner dimension
        """
        step = self._measurement.get_step_item(self.xy_selector.x)
        return len(step.calculate_values())

    def get_data(self, raw_data: dict[str, np.ndarray]) -> GraphData:
        """Get data for current view

        Args:
            raw_data (dict[str, np.ndarray]): dictionary of input data arrays

        Returns:
            GraphData: data for current view
        """
        data = GraphData(
            x_name=self.xy_selector.x,
            y_name=self.xy_selector.y,
            z_name=self.xy_selector.z,
            x_unit=self._measurement.get_channel(self.xy_selector.x).unit_physical,
            y_unit=self._measurement.get_channel(self.xy_selector.y).unit_physical,
        )
        data.y = raw_data[self.xy_selector.y]
        # convert data to list of entries, if needed
        if not self.use_entry_list:
            # re-order data to match selected x axis
            step_names = [step.name for step in self.get_active_step_items()]
            index_x = step_names.index(self._map_all_to_primary[self.xy_selector.x])
            data.y = np.moveaxis(data.y, index_x, 0)
            # create selection list matching data dimension
            selection = []
            cumulative_size = [1]
            for step in self.get_active_step_items(include_x=False):
                # use selection if available from column selector, else show all entries
                size = len(step.calculate_values())
                selection.append(self.selected_cols.get(step.name, list(range(size))))
                # cumulative size of data for each step item
                cumulative_size.append(size * cumulative_size[-1])
            # strip last element from cumulative size (=total size of data)
            cumulative_size = cumulative_size[:-1]
            # expanded selection to list with all column values
            all_values = [v.T.flatten() for v in np.meshgrid(*selection, indexing="ij")]
            # create list of selected rows by multiplying column with size
            self.selected_rows = np.dot(cumulative_size, all_values).tolist()

        if self.selected_rows == 0.0:
            # set selected rows to single list of element 0
            self.selected_rows = [0]

        if len(self.selected_rows) == 0:
            # clear data if no rows are selected
            data.y = np.array([])
            return data

        # make sure y data is at least 2D
        if len(data.y.shape) == 1:
            data.y = data.y.reshape((1, data.y.shape[0]))
        else:
            # transpose to make C-order (first dimension is outer step)
            data.y = data.y.T
        # for multi-dimensional data, convert to 2D to show list of data
        if len(data.y.shape) > 2:
            data.y = data.y.reshape((np.prod(data.y.shape[:-1]), data.y.shape[-1]))
        # for z data, always retrieve from step values for now
        if self.xy_selector.z in self._outer_step_values:
            data.z = self._outer_step_values[self.xy_selector.z]
        else:
            data.z = np.arange(self.get_number_of_entries())
        # get labels
        try:
            data.z_unit = self._measurement.get_channel(data.z_name).unit_physical
        except ValueError:
            data.z_unit = ""

        # apply row selection
        data.y = data.y[self.selected_rows, :]
        data.z = data.z[self.selected_rows]
        if self.xy_selector.x in raw_data:
            data.x = raw_data[self.xy_selector.x]
            if len(data.x.shape) == 2:
                data.x = data.x[self.selected_rows, :]
        else:
            # get x data from step item if no data available
            step_item_x = self._measurement.get_step_item(self.xy_selector.x)
            data.x = step_item_x.calculate_values()

        # pick complex view, if data is complex
        self.xy_selector.apply_complex_view(data)
        # apply data operations
        self.xy_selector.apply_operations(data)
        # update logview trace or image mode, if requested by data
        if data.image_plot is not None:
            self.image = data.image_plot
        return data


def get_view_path(identifier: str) -> Path:
    """Get default path for saving/loading view

    Parameters
    ----------
    identifier : str
        Log ID for loading view from disk

    Returns
    -------
    Path
        Path to view file
    """
    folder_path = Path("~/.svalbard/views").expanduser()
    # make sure folder exists
    if not folder_path.exists():
        folder_path.mkdir(parents=True)
    return folder_path / f"{identifier}.json"


def get_log_view(measurement: Measurement, identifier: str = None) -> LogView:
    """Create a LogView representing the measurement.

    If the identifier string is given, the function will try to load a saved
    view from disk.  If not, a default view will be generated.

    Parameters
    ----------
    measurement : Measurement
        Measurement object to view
    identifier : str, optional
        Log ID for loading view from disk, by default None

    Returns
    -------
    LogView
        Resulting LogView object
    """
    file_path = get_view_path(identifier)
    # generate default view if no identifier is given or path does not exist
    if identifier is None or not file_path.exists():
        view = LogView()
        view.initialize_view(measurement, set_defaults=True)
        return view
    # try to load view from disk
    with open(file_path, "r") as f:
        data = f.read()
    # initialize view with loaded data
    try:
        view = LogView.model_validate_json(data)
        view.initialize_view(measurement)
    except ValidationError as exc:
        # if validation fails, print error to console and generate default view
        print(f"Error loading view from {file_path}:\n{exc}")
        view = LogView()
        view.initialize_view(measurement, set_defaults=True)
    return view
