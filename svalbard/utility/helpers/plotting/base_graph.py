"""Base widgets for plotting"""

import matplotlib as mpl
import numpy as np
from aq_common import str_helper
from qtpy import QtCharts, QtCore, QtGui, QtWidgets
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor

from .. import ui_tools
from ..ui_tools import MAC
from .graph_items import SCALE_FACTOR, GraphItem, ImageItem


class BaseGraph(QtCharts.QChartView):
    # define signal for tracking graph position change, for position toolbar
    PositionChanged = QtCore.Signal()

    def __init__(self, use_open_gl=True, parent=None):
        """A base graph widget, to be subclassed for adding data.

        Parameters
        ----------
        use_open_gl : bool, optional
            Whether to use openGL, by default True
        parent : QObject, optional
            Parent object, by default None
        """
        super().__init__(parent)
        self.use_open_gl = use_open_gl
        self.dark_mode = ui_tools.is_dark_mode()
        # initialize data
        self.traces = dict()
        self.xlim = np.array([-0.01, 1.01])
        self.ylim = np.array([-0.01, 1.01])
        self._old_axes_config = {}
        self.x_label: str = ""
        self.y_label: str = ""
        # scaling and unit settings
        self.x_unit: str = ""
        self.y_unit: str = ""
        self.x_scaling = 1.0
        self.y_scaling = 1.0
        self.autoscale_x = True
        self.autoscale_y = True
        self.log_x = False
        self.log_y = False
        # z axes defined here, not used for xy trace graphs
        self.autoscale_z = True
        self.zlim = np.array([0.0, 1.0])
        self.z_label: str = ""
        self.z_unit: str = ""
        self.z_scaling = 1.0
        self.z_prefix: str = ""
        # labels and margins
        self.use_prefix = True
        self.x_prefix = ""
        self.y_prefix = ""
        self.x_margin = 0.020
        self.y_margin = 0.030
        # initialize chart widget
        self._init_chart()
        self.chart.plotAreaChanged.connect(self._on_plot_area_changed)

    def moveEvent(self, event):
        # generate event for other widgets to handle
        self.PositionChanged.emit()
        return super().moveEvent(event)

    def _on_plot_area_changed(self, qrect: QtCore.QRect):
        """Callback function for plot area changes, will redraw the image

        Parameters
        ----------
        qrect : QtCore.QRect
            Current plot ares dimensions
        """
        # re-implement in subclasses
        pass

    def _init_chart(self) -> None:
        """Initialize chart with default settings"""
        # general settings
        self.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        self.setOptimizationFlag(self.OptimizationFlag.DontAdjustForAntialiasing)
        # fix bug with touch events on macboooks
        self.viewport().setAttribute(QtCore.Qt.WA_AcceptTouchEvents, False)
        # chart style
        self.chart = self.chart()
        self.chart.legend().hide()
        self.chart.setPlotAreaBackgroundVisible(True)
        self.chart.setPlotAreaBackgroundBrush(QColor("#FFFFFF"))
        self.chart.setPlotAreaBackgroundPen(QColor("#888888"))
        self.chart.setBackgroundVisible(False)
        self.chart.layout().setContentsMargins(0, 0, 0, 0)
        # axes parameters
        self.x_axis = QtCharts.QValueAxis()
        self.x_axis.setLinePenColor(QColor("#888888"))
        pen = QtGui.QPen(Qt.PenStyle.DashLine)
        pen.setWidthF(1)
        pen.setColor(QColor("#DFDFDF"))
        self.x_axis.setGridLinePen(pen)
        self.x_axis.setTickCount(2)
        self.x_axis.setTickType(self.x_axis.TickType.TicksDynamic)
        self.x_axis.setLabelFormat("%.9g")
        self.chart.addAxis(self.x_axis, Qt.AlignmentFlag.AlignBottom)
        # same for y
        self.y_axis = QtCharts.QValueAxis()
        self.y_axis.setLinePenColor(QColor("#888888"))
        pen = QtGui.QPen(Qt.PenStyle.DashLine)
        pen.setWidthF(1)
        pen.setColor(QColor("#DFDFDF"))
        self.y_axis.setGridLinePen(pen)
        self.y_axis.setTickCount(2)
        self.y_axis.setTickType(self.y_axis.TickType.TicksDynamic)
        self.y_axis.setLabelFormat("%.9g ")
        self.chart.addAxis(self.y_axis, Qt.AlignmentFlag.AlignLeft)
        # create hidden axes for data in non-scaled units
        self.x_axis_data = QtCharts.QValueAxis()
        self.chart.addAxis(self.x_axis_data, Qt.AlignmentFlag.AlignBottom)
        self.x_axis_data.hide()
        self.y_axis_data = QtCharts.QValueAxis()
        self.chart.addAxis(self.y_axis_data, Qt.AlignmentFlag.AlignLeft)
        self.y_axis_data.hide()
        # ticks
        self.tick_x = mpl.ticker.MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10])
        self.tick_y = mpl.ticker.MaxNLocator(nbins=7, steps=[1, 2, 2.5, 5, 10])
        # set font
        f = QtGui.QFont("Arial", 14 if MAC else 10)
        self.chart.setTitleFont(f)
        self.x_axis.setTitleFont(f)
        self.x_axis.setLabelsFont(f)
        self.y_axis.setTitleFont(f)
        self.y_axis.setLabelsFont(f)
        # dark mode style changes
        if self.dark_mode:
            self.chart.setPlotAreaBackgroundBrush(QColor("#171717"))
            self.chart.setPlotAreaBackgroundPen(QColor("#888888"))
            self.x_axis.setLinePenColor(QColor("#888888"))
            self.x_axis.setGridLineColor(QColor("#444444"))
            self.y_axis.setLinePenColor(QColor("#888888"))
            self.y_axis.setGridLineColor(QColor("#444444"))
            self.x_axis.setTitleBrush(QColor("#DDDDDD"))
            self.y_axis.setTitleBrush(QColor("#DDDDDD"))
            self.x_axis.setLabelsBrush(QColor("#DDDDDD"))
            self.x_axis.setLabelsBrush(QColor("#DDDDDD"))
            self.y_axis.setLabelsBrush(QColor("#DDDDDD"))

    def update_ui_from_config(self) -> None:
        """Update UI from config"""
        # to be implemented in subclasses
        pass

    def update_config_from_ui(self) -> None:
        """Update config from UI"""
        # to be implemented in subclasses
        pass

    def copy_image_to_clipboard(self):
        """Copy graph to clipboard"""
        # get widget pixmap and put it on clipboard
        pixmap = self.grab()
        clipboard = QtWidgets.QApplication.clipboard()
        clipboard.setPixmap(pixmap)

    def is_point_outside_plot_area(self, x: float, y: float) -> bool:
        """Check if coordinate is outside plot area

        Parameters
        ----------
        x : float
            X coordinate
        y : float
            Y coordinate

        Returns
        -------
        bool
            Returns True is point is outside plot area
        """
        p = self.chart.plotArea()
        return x < p.left() or x > p.right() or y < p.top() or y > p.bottom()

    def map_pixel_to_axes(self, x: int, y: int) -> tuple[float, float]:
        """Map widget pixel coordinates to plot area coordinates

        Parameters
        ----------
        x : int
            X pixel position, widget units
        y : int
            Y pixel position, widget units

        Returns
        -------
        tuple[float, float]
            (x, y) tuple with coordinates in plot axes units
        """
        x = x - self.chart.plotArea().left()
        y = -(y - self.chart.plotArea().bottom())
        width = self.chart.plotArea().width()
        height = self.chart.plotArea().height()
        # calculate coordinates in plot area
        xa = self.xlim[0] + (self.xlim[1] - self.xlim[0]) * x / width
        ya = self.ylim[0] + (self.ylim[1] - self.ylim[0]) * y / height
        return (xa, ya)

    def map_axes_to_pixel(self, x: float, y: float) -> tuple[int, int]:
        """Map plot axes coordinates to widget pixel coordinates

        Parameters
        ----------
        x : float
            X-value, axes coordinate
        y : float
            Y-value, axes coordinate

        Returns
        -------
        tuple[int, int]
            (x, y) tuple with coordinates in widget pixel units
        """
        left = self.chart.plotArea().left()
        bottom = self.chart.plotArea().bottom()
        width = self.chart.plotArea().width()
        height = self.chart.plotArea().height()
        # calculate coordinates in widget pixel units
        xp = left + width * (x - self.xlim[0]) / (self.xlim[1] - self.xlim[0])
        yp = bottom - height * (y - self.ylim[0]) / (self.ylim[1] - self.ylim[0])
        return (xp, yp)

    def _get_axes_config(self) -> dict:
        """Get axes configuration as a dict, for caching redraws

        Returns
        -------
        dict
            Dict with axes configuration
        """
        d = {}
        # get limits
        self._calculate_data_scaling()
        d["xlim"] = list(self.xlim * self.x_scaling)
        d["ylim"] = list(self.ylim * self.y_scaling)
        # create axes labels
        if self.x_unit != "":
            d["xlabel"] = f"{self.x_label} ({self.x_prefix}{self.x_unit})"
        else:
            d["xlabel"] = self.x_label
        if self.y_unit != "":
            d["ylabel"] = f"{self.y_label} ({self.y_prefix}{self.y_unit})"
        else:
            d["ylabel"] = self.y_label
        if self.z_unit != "":
            d["zlabel"] = f"{self.z_label} ({self.z_prefix}{self.z_unit})"
        else:
            d["zlabel"] = self.z_label
        return d

    def redraw(self) -> bool:
        """Redraw graph

        Returns
        -------
        bool
            Returns True if axes were redrawn
        """
        # start by scaling graph axes based on autoscale settings
        if self.autoscale_x:
            self.scale_x()
        if self.autoscale_y:
            self.scale_y()
        # get axes limits and labels
        cfg = self._get_axes_config()
        if cfg == self._old_axes_config:
            # no change, don't redraw
            return False
        self._old_axes_config = cfg
        xlim, ylim = cfg["xlim"], cfg["ylim"]
        # turn off tick lables while drawing to avoid issues with slow updates
        self.x_axis.setLabelsVisible(False)
        self.y_axis.setLabelsVisible(False)
        # swith to fixed ticks while redrawing - seems not needed anymore
        # self.x_axis.setTickType(self.x_axis.TickType.TicksFixed)
        # self.y_axis.setTickType(self.y_axis.TickType.TicksFixed)
        # update tick to new values
        tick_values = self.tick_x.tick_values(*xlim)
        # fix bug with rounding errors close to zero by forcing tick at zero
        if xlim[0] < 0 < xlim[-1]:
            self.x_axis.setTickAnchor(0.0)
        else:
            self.x_axis.setTickAnchor(tick_values[0])
        self.x_axis.setTickInterval(tick_values[1] - tick_values[0])
        tick_values = self.tick_y.tick_values(*ylim)
        # fix bug with rounding errors close to zero by forcing tick at zero
        if ylim[0] < 0 < ylim[-1]:
            self.y_axis.setTickAnchor(0.0)
        else:
            self.y_axis.setTickAnchor(tick_values[0])
        self.y_axis.setTickInterval(tick_values[1] - tick_values[0])
        # set labels
        self.x_axis.setTitleText(cfg["xlabel"])
        self.y_axis.setTitleText(cfg["ylabel"])
        # set new range
        self.x_axis.setRange(xlim[0], xlim[1])
        self.x_axis_data.setRange(
            SCALE_FACTOR * self.xlim[0], SCALE_FACTOR * self.xlim[1]
        )
        self.y_axis.setRange(ylim[0], ylim[1])
        self.y_axis_data.setRange(
            SCALE_FACTOR * self.ylim[0], SCALE_FACTOR * self.ylim[1]
        )
        # turn on ticks and labels again
        # self.x_axis.setTickType(self.x_axis.TickType.TicksDynamic)
        # self.y_axis.setTickType(self.y_axis.TickType.TicksDynamic)
        self.x_axis.setLabelsVisible(True)
        self.y_axis.setLabelsVisible(True)
        # adding a repaint here seems to fix a bug with the tick labels not clearing
        self.repaint()
        # self.update() if WIN else self.repaint()
        return True

    def set_xlabel(self, label: str, unit: str = None) -> None:
        """Set x-axis label

        Parameters
        ----------
        label : str
            Label

        unit : str
            Unit, by default None
        """
        self.x_label = label
        if unit is not None:
            self.x_unit = unit

    def set_ylabel(self, label: str, unit: str = None) -> None:
        """Set y-axis label

        Parameters
        ----------
        label : str
            Label

        unit : str
            Unit, by default None
        """
        self.y_label = label
        if unit is not None:
            self.y_unit = unit

    def set_zlabel(self, label: str, unit: str = None) -> None:
        """Set z-axis label

        Parameters
        ----------
        label : str
            Label

        unit : str
            Unit, by default None
        """
        self.z_label = label
        if unit is not None:
            self.z_unit = unit

    def _calculate_data_scaling(self) -> None:
        """Calculate scaling for graph data"""
        # check if units and data are available, if not don't scale
        if not self.use_prefix or self.x_unit == "":
            self.x_scaling = 1.0
            self.x_prefix = ""
        else:
            # unit available, scale to prefix
            x_max = np.max(np.abs(self.xlim))
            x_max *= 1 + self.x_margin
            si_value = str_helper.convert_to_si(x_max)
            self.x_scaling = si_value["scale"]
            self.x_prefix = si_value["prefix"]

        # same for y
        if not self.use_prefix or self.y_unit == "":
            self.y_scaling = 1.0
            self.y_prefix = ""
        else:
            y_max = np.max(np.abs(self.ylim))
            y_max *= 1 + self.y_margin
            si_value = str_helper.convert_to_si(y_max)
            self.y_scaling = si_value["scale"]
            self.y_prefix = si_value["prefix"]

        # for z, scaling should be based on full-range
        if not self.use_prefix or self.z_unit == "":
            self.z_scaling = 1.0
            self.z_prefix = ""
        else:
            # unit available, scale to prefix
            z_range = self._get_z_data_range()
            # make sure there is data
            if len(z_range) == 2:
                z_max = np.max(np.abs(z_range))
                si_value = str_helper.convert_to_si(z_max)
                self.z_scaling = si_value["scale"]
                self.z_prefix = si_value["prefix"]

    def _set_axes_labels(self) -> None:
        """Update labels with correct description and units"""
        # create labels
        if self.x_unit != "":
            xlabel = f"{self.x_label} ({self.x_prefix}{self.x_unit})"
        else:
            xlabel = self.x_label
        if self.y_unit != "":
            ylabel = f"{self.y_label} ({self.y_prefix}{self.y_unit})"
        else:
            ylabel = self.y_label
        # set labels
        self.x_axis.setTitleText(xlabel)
        self.y_axis.setTitleText(ylabel)

    def _get_all_graph_items(self) -> list[GraphItem]:
        """Get all graph items in graph

        Returns
        -------
        list[GraphItem]
            List of all graph items
        """
        # subclasses will add more items
        return []

    def _get_x_data_range(self) -> np.ndarray:
        """Get x range from data

        Returns
        -------
        np.ndarray
            Two-element array with min and max x values
        """
        # get list of all graph items
        items = self._get_all_graph_items()
        # return empty list if no data
        if len(items) == 0:
            return []
        min_x = np.nanmin(list([item.min_x for item in items]))  # use list for nanmin
        max_x = np.nanmax(list([item.max_x for item in items]))  # use list for nanmax
        if np.isnan(min_x) or np.isnan(max_x):
            return []
        return np.array([min_x, max_x], dtype=float)

    def scale_x(self) -> None:
        """Autoscale x-axis based on data"""
        # get max/min
        x_range = self._get_x_data_range()
        # do nothing if no data
        if len(x_range) < 2:
            return
        # add margins
        x_range = x_range.mean() + (x_range - x_range.mean()) * (1.0 + self.x_margin)
        # make sure range isn't empty
        if x_range[0] == x_range[1]:
            if x_range[0] == 0.0:
                x_range = [-1.08, 1.08]
            else:
                x_range *= [0.988, 1.012]
        # set range
        self.xlim = np.array(x_range)
        # enable autoscaling
        self.autoscale_x = True
        self.update_ui_from_config()

    def set_xlim(self, xlim) -> None:
        """Set x-axis limits

        Parameters
        ----------
        xlim : tuple
            Limits
        """
        # make sure range is finite
        if np.all(np.isfinite(xlim)):
            self.xlim = np.array(xlim)
            # disable autoscaling
            self.autoscale_x = False
            self.update_ui_from_config()

    def _get_y_data_range(self) -> np.ndarray:
        """Get y range from data

        Returns
        -------
        np.ndarray
            Two-element array with min and max y values
        """
        # get list of all graph items
        items = self._get_all_graph_items()
        # return empty list if no data
        if len(items) == 0:
            return []
        min_y = np.nanmin(list([item.min_y for item in items]))  # use list for nanmin
        max_y = np.nanmax(list([item.max_y for item in items]))  # use list for nanmax
        if np.isnan(min_y) or np.isnan(max_y):
            return []
        return np.array([min_y, max_y], dtype=float)

    def scale_y(self) -> None:
        """Autoscale y-axis based on data"""
        # get max/min
        y_range = self._get_y_data_range()
        # do nothing if no data
        if len(y_range) < 2:
            return
        # add margins
        y_range = y_range.mean() + (y_range - y_range.mean()) * (1.0 + self.y_margin)
        # make sure range isn't empty
        if y_range[0] == y_range[1]:
            if y_range[0] == 0.0:
                y_range = [-1.08, 1.08]
            else:
                y_range *= [0.988, 1.012]
        # set range
        self.ylim = np.array(y_range)
        # enable autoscaling
        self.autoscale_y = True
        self.update_ui_from_config()

    def set_ylim(self, ylim) -> None:
        """Set Y-axis limits

        Parameters
        ----------
        ylim : tuple
            Limits
        """
        # make sure range is finite
        if np.all(np.isfinite(ylim)):
            self.ylim = np.array(ylim)
            # disable autoscaling
            self.autoscale_y = False
            self.update_ui_from_config()

    def _get_z_data_range(self) -> np.ndarray:
        """Get z range from data

        Returns
        -------
        np.ndarray
            Two-element array with min and max z values
        """
        # get list of all graph items
        items = self._get_all_graph_items()
        # only include image items
        items = [item for item in items if isinstance(item, ImageItem)]
        # return empty list if no data
        if len(items) == 0:
            return []
        min_z = np.nanmin(list([item.min_z for item in items]))  # use list for nanmin
        max_z = np.nanmax(list([item.max_z for item in items]))  # use list for nanmax
        if np.isnan(min_z) or np.isnan(max_z):
            return []
        return np.array([min_z, max_z], dtype=float)

    def scale_z(self) -> None:
        """Autoscale z-axis based on data"""
        # get max/min
        z_range = self._get_z_data_range()
        # do nothing if no data
        if len(z_range) < 2:
            return
        # make sure range isn't empty
        if z_range[0] == z_range[1]:
            z_range = [0.0, 1.0]
        # set range
        self.zlim = np.array(z_range)
        # enable autoscaling
        self.autoscale_z = True
        self.update_ui_from_config()

    def set_zlim(self, zlim) -> None:
        """Set z-axis limits

        Parameters
        ----------
        zlim : tuple
            Limits
        """
        # make sure range is finite
        if np.all(np.isfinite(zlim)) and len(zlim) == 2:
            self.zlim = np.array(zlim)
            # disable autoscaling
            self.autoscale_z = False
            self.update_ui_from_config()

    def scale_xy(self) -> None:
        """Autoscale axes of graph based on data range"""
        self.scale_x()
        self.scale_y()

    def clear_all(self) -> None:
        """Remove all data"""
        self.chart.removeAllSeries()
        # reset limits and scaling
        self.xlim = np.array([-0.01, 1.01])
        self.ylim = np.array([-0.01, 1.01])
        self.ylim = np.array([0.0, 1.0])
        self.x_scaling = 1.0
        self.y_scaling = 1.0
        self.z_scaling = 1.0
        self.autoscale_x = True
        self.autoscale_y = True
        self.autoscale_z = True
