"""Widget for configuring data views"""

import functools
from datetime import datetime

import numpy as np
from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import QSize, Qt
from qtpy.QtWidgets import QSizePolicy, QWidget

from ...data_model.data_file import Measurement, MetaData
from ...data_model.measurement.channel import Channel
from ...data_model.measurement.step_item import StepItem
from ...utility import str_helper
from ..models import data_view
from ..models.data_view import ComplexView, DataOperations, LogView, XYSelector
from . import ui_controls, ui_tools, ui_widgets
from .ui_tools import MAC


class XYGroupBox(QtWidgets.QGroupBox, ui_controls.WidgetInterface):
    """Group box for setting XY channels."""

    def __init__(self):
        super().__init__()
        # create layout for groupbox
        layout_x = QtWidgets.QGridLayout()
        layout_x.setContentsMargins(6, 6, 6, 6)
        layout_x.setVerticalSpacing(5)

        self.combo_y = ui_widgets.ComboBox()
        ui_tools.set_tip(self.combo_y, "Data for y axis")
        self.combo_x = ui_widgets.ComboBox()
        ui_tools.set_tip(self.combo_x, "Data for x axis")
        self.combo_z = ui_widgets.ComboBox()
        ui_tools.set_tip(self.combo_x, "Data for 3rd dimension")
        # simple data processing
        self.combo_data_ops = ui_controls.EnumControl(DataOperations)
        self.combo_complex = ui_controls.EnumControl(ComplexView)
        self.combo_complex.setMaximumWidth(120)
        self.combo_complex.set_value(ComplexView.MAG)
        self.int_smooth = ui_controls.IntControl()
        self.int_smooth.setMaximumWidth(60)
        self.int_smooth.setRange(1, 999)
        self.int_smooth.setValue(1)
        self.int_bins = ui_controls.IntControl()
        self.int_bins.setMaximumWidth(60)
        self.int_bins.setRange(2, 999)
        self.int_bins.setValue(51)
        self.int_bins.setHidden(True)
        self.label_smooth = QtWidgets.QLabel("Smooth:")
        # complex controls
        self.float_delay = ui_controls.FloatControl()
        self.float_rotation = ui_controls.FloatControl()
        self.check_auto_rotate = ui_controls.BoolControl("Auto-rotate:")
        self.check_auto_delay = ui_controls.BoolControl("Auto-delay:")
        self.check_plot_db = ui_controls.BoolControl("Plot in dB")
        self.check_unwrap = ui_controls.BoolControl("Unwrap angle")
        # complex layout
        self.layout_complex = QtWidgets.QGridLayout()
        self.layout_complex.setContentsMargins(0, 0, 0, 0)
        self.layout_complex.setHorizontalSpacing(6)
        self.layout_complex.setVerticalSpacing(3)
        self.layout_complex.addWidget(self.check_auto_rotate, 0, 0)
        self.layout_complex.addWidget(self.float_rotation, 1, 0)
        # create vertical spacers
        vertical_1 = QtWidgets.QFrame()
        vertical_1.setFrameShape(QtWidgets.QFrame.VLine)
        vertical_1.setFrameShadow(QtWidgets.QFrame.Sunken)
        vertical_2 = QtWidgets.QFrame()
        vertical_2.setFrameShape(QtWidgets.QFrame.VLine)
        vertical_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.layout_complex.addWidget(vertical_1, 0, 1, 2, 1)
        self.layout_complex.addWidget(self.check_auto_delay, 0, 2)
        self.layout_complex.addWidget(self.float_delay, 1, 2)
        self.layout_complex.addWidget(vertical_2, 0, 3, 2, 1)
        self.layout_complex.addWidget(self.check_unwrap, 0, 4)
        self.layout_complex.addWidget(self.check_plot_db, 1, 4)
        self.layout_complex.setColumnStretch(0, 100)
        self.layout_complex.setColumnStretch(1, 0)
        self.layout_complex.setColumnStretch(2, 100)
        self.layout_complex.setColumnStretch(3, 0)
        self.layout_complex.setColumnStretch(4, 100)
        self.widget_complex = QtWidgets.QWidget()
        self.widget_complex.setLayout(self.layout_complex)
        # layout channel selector ctrls
        layout_xy = QtWidgets.QGridLayout()
        layout_xy.setContentsMargins(6, 6, 6, 4)
        layout_xy.setHorizontalSpacing(6)
        layout_xy.setVerticalSpacing(3)
        # layout for y channel selector, needed for hiding complex controls
        layout_y = QtWidgets.QHBoxLayout()
        layout_y.setContentsMargins(0, 0, 0, 0)
        layout_y.setSpacing(0)
        layout_y.addWidget(self.combo_y)
        layout_y.addWidget(self.combo_complex)
        layout_y.setSizeConstraint(QtWidgets.QHBoxLayout.SetMinimumSize)
        layout_xy.addWidget(QtWidgets.QLabel("Y:"), 0, 0)
        layout_xy.addLayout(layout_y, 0, 1, 1, 5)
        layout_xy.addWidget(QtWidgets.QLabel("X:"), 1, 0)
        layout_xy.addWidget(self.combo_x, 1, 1, 1, 5)
        layout_xy.addWidget(self.widget_complex, 2, 0, 1, 6)
        layout_xy.addWidget(QtWidgets.QLabel("Operation:"), 3, 0, 1, 2)
        layout_xy.addWidget(self.combo_data_ops, 3, 2, 1, 2)
        layout_xy.addWidget(self.label_smooth, 3, 4)
        layout_xy.addWidget(self.int_bins, 3, 5)
        layout_xy.addWidget(self.int_smooth, 3, 5)
        layout_xy.addWidget(QtWidgets.QLabel("Z:"), 4, 0)
        layout_xy.addWidget(self.combo_z, 4, 1, 1, 5)
        layout_xy.setColumnStretch(0, 0)
        layout_xy.setColumnStretch(1, 10)
        layout_xy.setColumnStretch(2, 100)
        layout_xy.setColumnStretch(3, 0)
        layout_xy.setColumnStretch(4, 0)
        layout_xy.setColumnStretch(5, 30)
        # add widget to group box
        self.setLayout(layout_xy)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self.setTitle("Data selection")
        # store individual widgets in dict, for easy access
        self.widgets = {
            "complex_view": self.combo_complex,
            "auto_rotate": self.check_auto_rotate,
            "rotation": self.float_rotation,
            "auto_delay": self.check_auto_delay,
            "delay": self.float_delay,
            "unwrap": self.check_unwrap,
            "plot_db": self.check_plot_db,
            "operation": self.combo_data_ops,
            "bins": self.int_bins,
            "smooth": self.int_smooth,
        }
        # make sure widget are enabled according to current state
        self.set_callback(self.update_state, with_value=False)

    def get_value(self) -> XYSelector:
        # re-implementation of WidgetInteface.get_value
        # special case for x/y combo boxes, treat as strings
        xy = XYSelector(
            x=self.combo_x.currentText(),
            y=self.combo_y.currentText(),
            z=self.combo_z.currentText(),
        )
        # for other widgets, set attribute of xy selector
        for key, widget in self.widgets.items():
            setattr(xy, key, widget.get_value())
        return xy

    def set_value(self, value: XYSelector, block_signals=True) -> None:
        # re-implementation of WidgetInteface.set_value
        # treat x/y channels as strings
        def set_combo_text(combo, value):
            combo.blockSignals(True)
            combo.setCurrentText(value)
            combo.blockSignals(False)

        set_combo_text(self.combo_y, value.y)
        set_combo_text(self.combo_x, value.x)
        set_combo_text(self.combo_z, value.z)
        # for all other, call set_value on controls
        for key, widget in self.widgets.items():
            widget.set_value(getattr(value, key), block_signals=block_signals)

    def set_callback(self, callback: callable, with_value=True) -> None:
        # re-implementation of WidgetInteface.set_callback
        if with_value:
            callback = functools.partial(self._callback_with_value, callback)
        for widget in self.widgets.values():
            widget.set_callback(callback, with_value=False)
        # add callbacks for x/y combo boxes
        self.combo_y.currentIndexChanged[int].connect(callback)
        self.combo_x.currentIndexChanged[int].connect(callback)
        self.combo_z.currentIndexChanged[int].connect(callback)

    def show_complex_widgets(self, show: bool = False) -> None:
        """Show/hide complex widgets based on input argument"""
        self.widget_complex.setVisible(show)
        self.combo_complex.setVisible(show)

    def update_state(self) -> None:
        """Update state of UI widgets to match configuration state"""
        # get current state
        state = self.get_value()
        self.float_rotation.setEnabled(not state.auto_rotate)
        self.float_delay.setEnabled(not state.auto_delay)
        # self.check_unwrap.setEnabled(state.complex == ComplexView.PHASE)
        # self.check_plot_db.setEnabled(state.complex == ComplexView.MAG)
        is_histogram = state.operation in (
            DataOperations.HISTOGRAM,
            DataOperations.HISTOGRAM_IQ,
        )
        self.label_smooth.setText("Bins" if is_histogram else "Smooth")
        self.int_smooth.setVisible(not is_histogram)
        self.int_bins.setVisible(is_histogram)
        # for now, disable selection of x/y channels
        # self.combo_x.setEnabled(False)
        # self.combo_y.setEnabled(False)

    def set_labels(self, x_labels: list[str], y_labels: list[str], z_labels: list[str]):
        """Set labels for x/y combo boxes

        Parameters
        ----------
        labels : list[str]
            List of labels to set
        """
        for combo, labels in zip(
            (self.combo_x, self.combo_y, self.combo_z), (x_labels, y_labels, z_labels)
        ):
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(labels)
            combo.blockSignals(False)


class LogViewWidget(QWidget, ui_controls.WidgetInterface):
    """Widget allowing user to control settings of a LogView object.

    Parameters
    ----------
    metadata : MetaData
        MetaData object use to popluate channel names and log entries
    """

    def __init__(self, metadata: MetaData):
        super().__init__()
        # store measurement object
        self._metadata = metadata
        # create default log view, needed for list model initialization
        self.log_view = data_view.get_log_view(self._metadata.measurement)
        # create subwidgets
        self.xy_config = XYGroupBox()
        self.xy_config.set_callback(self._on_xy_config, with_value=False)

        # entry list
        self.entry_list = ui_widgets.ListSelectionView(multi_selection=True)
        # create new selection model for log list and set callback
        model = LogListModel(self.log_view, self._metadata, None)
        self.entry_list.setModel(model)
        self.entry_list.selectionModel().selectionChanged.connect(
            self.on_entry_selection_changed
        )
        # resize columns to contents based on new model, then add some extra space
        for n in range(self.entry_list.model().columnCount() - 1):
            self.entry_list.resizeColumnToContents(n)
            n_pixel = self.entry_list.columnWidth(n)
            self.entry_list.setColumnWidth(n, n_pixel + 9)
        # guess column width for data columns
        self.entry_list.horizontalHeader().setStretchLastSection(True)
        # selection controls
        self.label_selection = QtWidgets.QLabel("Selection: 0/100")
        self.push_select_all = QtWidgets.QPushButton("Select all")
        self.push_select_all.setMaximumWidth(120)
        self.push_select_all.clicked.connect(self.entry_list.selectAll)
        # create layout for list widget
        layout_list_selector = QtWidgets.QGridLayout()
        layout_list_selector.setContentsMargins(0, 0, 0, 0)
        layout_list_selector.addWidget(self.entry_list, 0, 0, 1, 2)
        layout_list_selector.addWidget(self.label_selection, 1, 0)
        layout_list_selector.addWidget(self.push_select_all, 1, 1)
        widget_list_selector = QWidget()
        widget_list_selector.setLayout(layout_list_selector)

        # column selector
        self.widget_step_selector = ColumnSelectorWidget(metadata.measurement)
        self.widget_step_selector.set_callback(self._on_new_column_selection)
        # create stacked widget for holding both
        self.stack_selector = QtWidgets.QStackedWidget()
        self.stack_selector.addWidget(widget_list_selector)
        self.stack_selector.addWidget(self.widget_step_selector)

        # create actions for toobars, menus, etc
        self._create_actions()
        # create layout
        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.xy_config, 0, 0, 1, 2)
        layout.addWidget(self.stack_selector, 2, 0, 1, 2)
        layout.setVerticalSpacing(6)
        layout.setContentsMargins(6, 6, 6, 6)
        self.setLayout(layout)
        self.setContentsMargins(0, 0, 0, 0)
        # update UI values from view
        self.set_value(self.log_view)

    def _on_new_column_selection(self):
        # callback for new column selection
        pass

    def _on_xy_config(self):
        # callback for new X/Y configuration
        # save previous y/x selection before retrieving with new UI values
        old_y_selection = self.log_view.xy_selector.y
        old_x_names = self.log_view.get_x_channel_names()
        # get value of complete UI to make sure internal log_view is up to date
        self.get_value()
        # check if y selector changed in UI
        if old_y_selection != self.log_view.xy_selector.y:
            # y changed, set default x if default was not available earlier
            new_default_x = self.log_view.get_default_x_selection()
            if new_default_x not in old_x_names:
                self.log_view.xy_selector.x = new_default_x
                # clear entry selection to default if x selection changes
                self.log_view.select_default_entries()
        # validate log view based on new settings
        self.log_view.validate_xy_selector()
        # update column selector to show the right number of columns
        self.entry_list.model().update_log_view(self.log_view)
        self.entry_list.model().layoutChanged.emit()
        # resize columns to contents based on new model, then add some extra space
        for n in range(self.entry_list.model().columnCount() - 1):
            self.entry_list.resizeColumnToContents(n)
            n_pixel = self.entry_list.columnWidth(n)
            self.entry_list.setColumnWidth(n, n_pixel + 9)
        # guess column width for data columns
        self.entry_list.horizontalHeader().setStretchLastSection(True)
        # log_view settings may have been updated during validation, force UI update
        self.set_value(self.log_view, block_signals=True)
        # update state of other UI widgets
        self.update_state()

    def _create_actions(self):
        """Create actions for menu, toolbars and context menus"""
        # create actions
        self.action_image = ui_tools.create_action(
            self,
            "Show image",
            self.update_state,
            checkable=True,
            tip="Show as image",
            icon=ui_tools.get_theme_icon("graph-image"),
        )
        # action for switching between entry list and column selector view
        self.action_use_entry_list = ui_tools.create_action(
            self,
            "Select Data by Log Entry",
            self.update_state,
            checkable=True,
        )
        self.action_use_entry_list.setChecked(True)

    def on_entry_selection_changed(self):
        """Callback for selection change in entry list"""
        n_selected = len(self.entry_list.get_selected_rows())
        n_all = self.entry_list.model().rowCount()
        self.label_selection.setText(f"Selection: {n_selected}/{n_all}")

    def get_value(self) -> LogView:
        # re-implementation of WidgetInteface.get_value
        # start with default view, then updating values from UI widgets
        self.log_view.xy_selector = self.xy_config.get_value()
        self.log_view.use_entry_list = self.action_use_entry_list.isChecked()
        self.log_view.selected_rows = self.entry_list.get_selected_rows()
        self.log_view.selected_cols = self.widget_step_selector.get_value()
        self.log_view.image = self.action_image.isChecked()
        return self.log_view

    def set_value(self, value: LogView, block_signals=True) -> None:
        # re-implementation of WidgetInteface.set_value
        self.log_view = value
        # updating comboboxes with available channels
        self.xy_config.set_labels(
            self.log_view.get_x_channel_names(),
            self.log_view.get_y_channel_names(),
            self.log_view.get_z_channel_names(),
        )
        # set xy selector and image mode
        self.xy_config.set_value(value.xy_selector, block_signals=block_signals)
        self.action_image.blockSignals(True)
        self.action_image.setChecked(self.log_view.image)
        self.action_image.blockSignals(False)
        self.action_use_entry_list.blockSignals(True)
        self.action_use_entry_list.setChecked(self.log_view.use_entry_list)
        self.action_use_entry_list.blockSignals(False)
        # set selected entries
        self.entry_list.select_rows(value.selected_rows)
        self.widget_step_selector.set_value(self.log_view.selected_cols)
        # make model signal that the list selection has changed
        self.entry_list.model().layoutChanged.emit()
        # update state of UI widgets
        self.update_state()

    def update_state(self):
        """Update state of UI widgets to match configuration state"""
        log_view = self.get_value()
        # show list or column selector
        self.stack_selector.setCurrentIndex(0 if log_view.use_entry_list else 1)
        # update selection controls/labels
        if log_view.use_entry_list:
            self.on_entry_selection_changed()
        else:
            self.widget_step_selector.update_state(log_view)

    def set_callback(self, callback: callable, with_value=True) -> None:
        # re-implementation of WidgetInteface.set_callback
        self.xy_config.set_callback(callback, with_value=with_value)
        self.entry_list.selectionModel().selectionChanged.connect(callback)
        self.widget_step_selector.set_callback(callback)
        self.action_image.triggered.connect(callback)
        self.action_use_entry_list.triggered.connect(callback)

    def update_calculated_values(self, log_view: LogView) -> None:
        """Update calculated values in log view widget (e.g. phase rotations)

        Parameters
        ----------
        log_view : LogView
            Log view object with updated calculated values
        """
        # update just xy selector, and block signals
        self.xy_config.set_value(log_view.xy_selector, block_signals=True)
        # update image mode, may have changed depending on view (for IQ histograms)
        self.action_image.blockSignals(True)
        self.action_image.setChecked(log_view.image)
        self.action_image.blockSignals(False)
        # show/hide complex widgets
        self.xy_config.show_complex_widgets(log_view.xy_selector.is_data_complex())


class ColumnSelectorWidget(QWidget, ui_controls.WidgetInterface):
    """Widget for column data selection.

    Parameters
    ----------
    metadata : MetaData
        MetaData object use to popluate column values
    """

    def __init__(self, measurement: Measurement):
        super().__init__()
        self._measurement = measurement
        # keep track of step item names corresponding to X/Z selection
        self._allowed_names: list[str] = []
        # column selector, key is step name
        self.column_widgets: dict[str, SingleColumnWidget] = {}
        layout = QtWidgets.QHBoxLayout()
        for n, step in enumerate(data_view.get_primary_step_items(measurement)):
            channel = measurement.get_channel(step.name)
            widget = SingleColumnWidget(step, channel)
            layout.addWidget(widget)
            self.column_widgets[step.name] = widget
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

    def get_value(self) -> dict[str, list[int]]:
        # re-implementation of WidgetInteface.get_value
        return {
            step_name: widget.get_value()
            for step_name, widget in self.column_widgets.items()
            if step_name in self._allowed_names
        }

    def set_value(self, value: dict[str, list[int]], block_signals=True) -> None:
        # re-implementation of WidgetInteface.set_value
        # input is dict with list of selected rows for each step name
        # go through all widgets and update all
        for step_name, widget in self.column_widgets.items():
            if step_name in value:
                widget.set_value(value[step_name])
            else:
                # reset selection to single entry if not visible
                widget.set_value([0])

    def set_callback(self, callback: callable, with_value=True) -> None:
        # re-implementation of WidgetInteface.set_callback
        # same callback for all selectors
        for n, widget in enumerate(self.column_widgets.values()):
            widget.set_callback(callback)

    def update_state(self, log_view: LogView) -> None:
        """Update view based on log view object"""
        #  get allowed steps items for current y selection
        self._allowed_names = [
            step.name
            for step in log_view.get_active_step_items(include_x=False, include_z=False)
        ]
        # show/hide for column widgets pending on step items visible in log view
        for step_name, column_widget in self.column_widgets.items():
            column_widget.setVisible(step_name in self._allowed_names)
            column_widget.on_entry_selection_changed()


class SingleColumnWidget(QWidget, ui_controls.WidgetInterface):
    """Widget for single column data selection.

    Parameters
    ----------
    metadata : MetaData
        MetaData object use to popluate column values
    """

    def __init__(self, step: StepItem, channel: Channel):
        super().__init__()
        layout = QtWidgets.QGridLayout()
        # create column list and model
        self.column_widget = ui_widgets.ListSelectionView(multi_selection=True)
        model = ColumnSelectionModel(step, channel)
        self.column_widget.setModel(model)
        self.column_widget.selectionModel().selectionChanged.connect(
            self.on_entry_selection_changed
        )
        layout.addWidget(self.column_widget, 0, 0, 1, 2)
        self.label_selection = QtWidgets.QLabel("0/100")
        push_select_all = QtWidgets.QPushButton("Select all")
        push_select_all.setMaximumWidth(80)
        push_select_all.clicked.connect(self.column_widget.selectAll)
        layout.addWidget(self.label_selection, 1, 0)
        layout.addWidget(push_select_all, 1, 1)
        layout.setVerticalSpacing(6)
        layout.setContentsMargins(0, 6, 6, 6)
        self.setLayout(layout)

    def get_value(self) -> list[int]:
        # re-implementation of WidgetInteface.get_value
        return self.column_widget.get_selected_rows()

    def set_value(self, value: list[int], block_signals=True) -> None:
        # set selected entries and tell model that the  selection changed
        self.column_widget.select_rows(value)
        self.column_widget.model().layoutChanged.emit()
        # update selection label
        self.on_entry_selection_changed()

    def set_callback(self, callback: callable, with_value=True) -> None:
        # re-implementation of WidgetInteface.set_callback
        self.column_widget.selectionModel().selectionChanged.connect(callback)

    def on_entry_selection_changed(self):
        """Callback for selection change in entry list"""
        n_selected = len(self.column_widget.get_selected_rows())
        n_all = self.column_widget.model().rowCount()
        self.label_selection.setText(f"{n_selected}/{n_all}")


class BaseTableModel(QtCore.QAbstractTableModel):
    """Model with proper rendering of header data on MacOS"""

    def __init__(self, parent=None):
        super().__init__(parent)
        # create empty header
        self._header = []

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return self._header[section]
        elif role == Qt.TextAlignmentRole:
            return Qt.AlignLeft
        elif MAC and role == Qt.SizeHintRole:
            size = QSize()
            size.setHeight(16)
            return size
        elif MAC and role == Qt.FontRole:
            font = QtGui.QFont()
            font.setPointSize(12)
            return font

    def columnCount(self, parent=None):
        return len(self._header)


class LogListModel(BaseTableModel):
    def __init__(self, log_view: LogView, metadata, data, parent=None):
        super().__init__(parent)
        self._data = data
        self._metadata = metadata
        self.update_log_view(log_view)

    def update_log_view(self, log_view: LogView):
        """Update table with new log view object

        Parameters
        ----------
        log_view : LogView
            New LogView object
        """
        self.log_view = log_view
        # pre-calculate values for each step item
        step_items = self.log_view.get_active_step_items(
            include_x=False, include_secondary=True
        )
        # generate list of channels for each step
        self.step_channels = [
            self._metadata.measurement.get_channel(step.name) for step in step_items
        ]
        # create header
        self._header = ["#", "Date", "Time", "# pts."]
        self._header.extend([step.name for step in step_items])
        # cache size, to avoid recalculation later in case log view changes
        self._number_of_entries = self.log_view.get_number_of_entries()

    def rowCount(self, parent=None):
        return self._number_of_entries

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            row = index.row()
            if 0 <= row < self.rowCount():
                column = index.column()

                # entry number
                if column == 0:
                    return str(row + 1)

                # get time stamp
                if column == 1 or column == 2:
                    time_db = self._metadata.date
                    if time_db is None:
                        return ""
                    if isinstance(time_db, datetime):
                        time_stamp = time_db
                    elif isinstance(time_db, str):
                        time_stamp = datetime.fromisoformat(time_db)
                    if column == 1:
                        return time_stamp.strftime("%Y-%m-%d")
                    return time_stamp.strftime("%H:%M:%S")

                # size, use step dimension for now
                if column == 3:
                    # np.count_nonzero(np.isnan(data))
                    return str(self.log_view.get_inner_dimension())

                # step values, get from pre-calculated list
                if column < self.columnCount():
                    channel = self.step_channels[column - 4]
                    value = self.log_view.get_outer_step_values(channel.name)[row]
                    # value = self.step_values[column - 4][row]
                    if isinstance(value, list) or isinstance(value, np.ndarray):
                        return str(value)
                    return str_helper.get_si_string(
                        value, channel.unit_physical, decimals=6
                    )


class ColumnSelectionModel(BaseTableModel):
    def __init__(self, step_item: StepItem, channel: Channel, parent=None):
        super().__init__(parent)
        self.step_item = step_item
        self.channel = channel
        # pre-calculate values for fast access
        self._step_values = step_item.calculate_values()
        # create header
        self._header = [step_item.name]

    def rowCount(self, parent=None):
        return len(self._step_values)

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            row = index.row()
            if 0 <= row < self.rowCount():
                row_value = self._step_values[row]
                if isinstance(row_value, list) or isinstance(row_value, np.ndarray):
                    return str(self._step_values[row])
                return str_helper.get_si_string(
                    self._step_values[row], self.channel.unit_physical, decimals=6
                )
