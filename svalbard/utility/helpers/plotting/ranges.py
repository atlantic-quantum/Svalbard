"""Widgets for plotting"""

from qtpy import QtCore, QtWidgets
from qtpy.QtWidgets import QSizePolicy

from .. import ui_controls, ui_widgets


class _AxesRangeGroupBox(QtWidgets.QGroupBox):
    """Group box for setting graph axis range, for use in SetAxesRangeDialog

    Parameters
    ----------
    label : str
        Label for axis, should be X or Y
    """

    def __init__(self, label: str = "X"):
        super().__init__()
        # creat controls
        self.min_value = ui_controls.FloatControl(digits=6)
        self.max_value = ui_controls.FloatControl(digits=6)
        self.button_scale = QtWidgets.QPushButton(f"Autoscale {label}")
        # create layout for x
        layout_x = QtWidgets.QGridLayout()
        layout_x.setContentsMargins(6, 6, 6, 6)
        layout_x.setVerticalSpacing(5)
        # log xcale, x
        self.combo_log = ui_widgets.ComboBox()
        self.combo_log.addItems(["Linear", "Log"])

        layout_x.addWidget(QtWidgets.QLabel("Scale type: "), 0, 0)
        layout_x.addWidget(self.combo_log, 0, 1)

        layout_x.addWidget(QtWidgets.QLabel("Min value: "), 1, 0)
        layout_x.addWidget(self.min_value, 1, 1)
        layout_x.addWidget(QtWidgets.QLabel("Max value: "), 2, 0)
        layout_x.addWidget(self.max_value, 2, 1)
        layout_x.addWidget(self.button_scale, 3, 1)
        layout_x.setColumnStretch(0, 1)
        layout_x.setColumnStretch(1, 2)
        # create group box
        self.setLayout(layout_x)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self.setTitle(f"{label} axis")


class SetAxesRangeDialog(QtWidgets.QDialog):
    """Dialog for setting range of graph axes

    Parameters
    ----------
    graph : Graph
        Graph to which this dialog applies
    """

    def __init__(self, graph):
        super().__init__(graph)
        self.graph = graph
        # turn off autoscaling
        self.graph.autoscale_x = False
        self.graph.autoscale_y = False
        self.graph.update_ui_from_config()
        # create controls
        self.group_x = _AxesRangeGroupBox("X")
        # set values
        self.group_x.min_value.set_value(self.graph.xlim[0])
        self.group_x.max_value.set_value(self.graph.xlim[1])
        self.group_x.combo_log.setCurrentIndex(1 if self.graph.log_x else 0)
        # set callbacks
        self.group_x.min_value.set_callback(self.update_x_range)
        self.group_x.max_value.set_callback(self.update_x_range)
        self.group_x.button_scale.clicked.connect(self._on_scale_x)
        self.group_x.combo_log.currentIndexChanged.connect(self._on_log_x)
        # same for y
        self.group_y = _AxesRangeGroupBox("Y")
        # set values
        self.group_y.min_value.set_value(self.graph.ylim[0])
        self.group_y.max_value.set_value(self.graph.ylim[1])
        self.group_y.combo_log.setCurrentIndex(1 if self.graph.log_y else 0)
        # set callbacks
        self.group_y.min_value.set_callback(self.update_y_range)
        self.group_y.max_value.set_callback(self.update_y_range)
        self.group_y.button_scale.clicked.connect(self._on_scale_y)
        self.group_y.combo_log.currentIndexChanged.connect(self._on_log_y)
        # ok and cancel buttons
        button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok)
        button_ok = button_box.button(QtWidgets.QDialogButtonBox.Ok)
        button_ok.setText("Close")
        # create layout
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        layout.addWidget(self.group_x)
        layout.addWidget(self.group_y)
        layout.addStretch(1)
        layout.addWidget(button_box)
        self.setLayout(layout)
        # connect slots and signals
        button_box.accepted.connect(self.accept)
        # set window title and sizing
        self.setWindowTitle("Set axes range")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self.sizeHint = lambda: QtCore.QSize(250, 10)
        self.resize(self.sizeHint())
        self.setSizeGripEnabled(True)
        # select all previous text
        self.group_x.min_value.setFocus()
        self.group_x.min_value.selectAll()

    def update_x_range(self, _) -> None:
        """Update range for x"""
        self.graph.set_xlim(
            [self.group_x.min_value.get_value(), self.group_x.max_value.get_value()]
        )
        self.graph.redraw()

    def update_y_range(self, _) -> None:
        """Update range for y"""
        self.graph.set_ylim(
            [self.group_y.min_value.get_value(), self.group_y.max_value.get_value()]
        )
        self.graph.redraw()

    def _on_log_x(self) -> None:
        """Callback on log/lin x button"""
        pass

    def _on_log_y(self):
        """Callback on log/lin y button"""
        pass

    def _on_scale_x(self):
        """Callback for clicking on autoscale button"""
        self.graph.scale_x()
        self.graph.redraw()
        self.group_x.min_value.set_value(self.graph.xlim[0])
        self.group_x.max_value.set_value(self.graph.xlim[1])

    def _on_scale_y(self):
        """Callback for clicking on autoscale button"""
        self.graph.scale_y()
        self.graph.redraw()
        self.group_y.min_value.set_value(self.graph.ylim[0])
        self.group_y.max_value.set_value(self.graph.ylim[1])
