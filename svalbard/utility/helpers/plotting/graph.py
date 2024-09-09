"""Widgets for plotting"""

import numpy as np
from qtpy import QtCore, QtGui

from .. import ui_tools
from .base_graph_ui import BaseGraphUI
from .graph_items import GraphItem, XYItem

TRACE_COLORS = [
    "#EA4335",
    "#0071BC",
    "#000000",
    "#CC900F",
    "#7D2E8D",
    "#32A853",
    "#3CADDD",
    "#A1132E",
    "#666666",
    "#83622C",
]

TRACE_COLORS_DARK = [
    "#FF4444",
    "#5588FF",
    "#FFFFFF",
    "#22FF22",
    "#FFDD00",
    "#FF33FF",
    "#33FFFF",
    "#BBBBBB",
    "#888888",
    "#A37235",
]


class GraphXY(BaseGraphUI):
    def __init__(self, use_open_gl=True, parent=None):
        """A graph widget for plotting XY trace data

        Parameters
        ----------
        use_open_gl : bool, optional
            Whether to use openGL, by default True
        parent : QObject, optional
            Parent object, by default None
        """
        super().__init__(use_open_gl, parent)
        # initialize trace data
        self.traces = dict()
        self._selected_trace = None
        # create caption and label for trace selection
        self.create_caption()

    def create_caption(self):
        """Create graph items for caption showing trace selection"""
        # caption box
        self.caption_rect = self.chart.scene().addRect(
            QtCore.QRect(0, 0, 10, 0),
            QtGui.QPen("#888888" if self.dark_mode else "#888888"),
            QtGui.QBrush("#171717" if self.dark_mode else "#FFFFFF"),
        )
        # caption line representing trace
        p = QtGui.QPen("#FF0000")
        p.setWidthF(3.0)
        self.caption_line = self.chart.scene().addLine(0, 0, 1, 1, p)
        # caption label
        f = QtGui.QFont("Arial", 14 if ui_tools.MAC else 10)
        self.caption_label = self.chart.scene().addSimpleText("Test", f)
        self.caption_label.setBrush(
            QtGui.QBrush("#DDDDDD" if self.dark_mode else "#000000")
        )
        # set z-values and hide at start
        self.CAPTION_ITEMS = [self.caption_rect, self.caption_line, self.caption_label]
        for n, c in enumerate(self.CAPTION_ITEMS):
            c.setZValue(101 + n)
            c.setVisible(False)

    def update_caption(self, label: str, color: str | QtGui.QColor):
        """Update caption for trace selection

        Parameters
        ----------
        label : str
            Caption label
        color : str
            Caption color
        """
        self.caption_label.setText(label)
        p = self.caption_line.pen()
        p.setColor(QtGui.QColor(color))
        self.caption_line.setPen(p)

    def reposition_caption(self):
        """Re-position caption box to top right corner, taking size into account"""
        # re-position selection caption
        buf = 8
        line = 20
        height = 18
        width = self.caption_label.boundingRect().width() + line + 3 * buf
        left = self.chart.plotArea().right() - width
        top = self.chart.plotArea().top() - 19
        mid = top + height / 2 - 1
        self.caption_rect.setRect(left, top - 2, width, height)
        self.caption_line.setLine(left + buf, mid, left + buf + line, mid)
        self.caption_label.setPos(left + 2 * buf + line, top)

    def _on_plot_area_changed(self, qrect: QtCore.QRect):
        super()._on_plot_area_changed(qrect)
        # re-position selection caption
        self.reposition_caption()

    def _get_all_graph_items(self) -> list[GraphItem]:
        """Get all graph items in graph

        Returns
        -------
        list[GraphItem]
            List of all graph items
        """
        # overload base class to include traces
        parent_items = super()._get_all_graph_items()
        return parent_items + list(self.traces.values())

    def register_mouse_press(self, button, xp: int, yp: int):
        # call base class, return if handled
        if super().register_mouse_press(button, xp, yp):
            return True
        # event not handled by base, check if trace selected
        x, y = self.map_pixel_to_axes(xp, yp)
        # calculate one pixel step size to axes units
        x2, y2 = self.map_pixel_to_axes(xp + 1, yp + 1)
        dx = abs(x2 - x)
        dy = abs(y2 - y)
        # interpolate over a few points arond x, to catch steep traces
        xx = np.linspace(x - 4 * dx, x + 4 * dx, 21)
        # calculate y distance to all traces in axes units
        min_dy = [
            (
                np.nan
                if len(trace._x) == 0
                else np.nanmin(np.abs(np.interp(xx, trace._x, trace._y) - y))
            )
            for trace in self.traces.values()
        ]
        min_index = np.nanargmin(min_dy)
        if min_dy[min_index] <= 4 * dy:
            # highlight selected trace and store it
            xy_item = list(self.traces.values())[min_index]
            xy_item.highlight_trace(True)
            self._selected_trace = xy_item
            # update caption
            self.update_caption(xy_item.label, xy_item.color)
            # show caption
            for c in self.CAPTION_ITEMS:
                c.setVisible(True)
            # re-position caption
            self.reposition_caption()
        return True

    def release_mouse_press(self, x: int, y: int) -> None:
        # check if trace was selected
        if self._selected_trace:
            # reset line width and forget selection
            self._selected_trace.highlight_trace(False)
            self._selected_trace = None
            # hide caption
            for c in self.CAPTION_ITEMS:
                c.setVisible(False)
        else:
            # call base class
            super().release_mouse_press(x, y)

    def create_trace(
        self,
        x: np.ndarray,
        y: np.ndarray,
        trace_n: int = None,
        label: str = None,
        color=None,
        linewidth=None,
        marker=True,
        marker_size=4.5,
    ) -> None:
        """Create XY trace for graph

        Parameters
        ----------
        x : np.ndarray
            X-values
        y : np.ndarray
            Y-values
        trace_n : int, optional
            Trace index, by default None
        label : str, optional
            Label describing trace in caption, by default None
        color ; str, optional
            Color of trace, by default None (auto)
        linewidth : float, optional
            Line width of trace, by default None (auto)
        marker : bool, optional
            Whether to plot markers for each data point, by default True
        marker_size : int, optional
            Whether to plot markers for each data point, by default 4.5
        """
        # create new trace if no trace_n given
        if trace_n is None:
            if len(self.traces) == 0:
                trace_n = 0
            else:
                trace_n = 1 + max(self.traces.keys())
        # remove if trace already exists
        if trace_n in self.traces:
            trace = self.traces.pop(trace_n)
            trace.remove_from_graph()
        # create trace
        colors = TRACE_COLORS_DARK if self.dark_mode else TRACE_COLORS
        color = colors[trace_n % len(colors)] if color is None else color
        linewidth = 1.0 if linewidth is None else linewidth
        label = str(trace_n) if label is None else label
        trace = XYItem(x, y, color, linewidth, marker, marker_size, label=label)
        # store traces
        self.traces[trace_n] = trace
        # add to graph
        trace.attach_to_graph(self)

    def update_trace_data(
        self, trace_n: int, x: np.ndarray, y: np.ndarray, label: str = None
    ) -> None:
        """Update data for specified XY-trace

        Parameters
        ----------
        trace_n : int
            Trace to update
        x : np.ndarray, optional
            New x data
        y : np.ndarray, optional
            New y data
        label : str, optional
            Label describing trace in caption, by default None
        """
        # create if trace doesn't exists
        if trace_n not in self.traces:
            self.create_trace(x, y, trace_n, label)
            return
        trace = self.traces[trace_n]
        trace.update_values(x, y)
        if label is not None:
            trace.label = label

    def clear_traces(self) -> None:
        """Clear all traces, but keep other graph items, including cursors"""
        # remove traces
        for trace in self.traces.values():
            trace.remove_from_graph()
        self.traces.clear()

    def clear_all(self) -> None:
        # remove traces
        self.traces.clear()
        # call base class to remove rest
        super().clear_all()


class LiveGraphXY(GraphXY):
    def __init__(self, n_history=5, use_open_gl=True, parent=None):
        """XY graph with live data updates

        Parameters
        ----------
        n_history : int, optional
            Number of historic traces to show, by default 5
        use_open_gl : bool, optional
            Whether to use openGL, by default True
        parent : QObject, optional
            Parent object, by default None
        """
        super().__init__(use_open_gl=use_open_gl, parent=parent)
        self.n_history = n_history
        # create history traces
        for n in range(self.n_history):
            if self.dark_mode:
                fade = 230 - 140 * (self.n_history - n) / self.n_history
            else:
                fade = 80 + 140 * (self.n_history - n) / self.n_history
            color = f"#{int(fade):02x}{int(fade):02x}{int(fade):02x}"
            self.create_trace([], [], n, color=color, linewidth=1, marker=False)
        # create live trace
        self.create_trace(
            [], [], self.n_history, color="#EB4E3E", linewidth=1.8, marker_size=6
        )

    def update_live_data(self, x, y, label=None):
        """Update current live XY trace with new data

        Parameters
        ----------
        x : np.ndarray
            X-values
        y : np.ndarray
            Y-values
        label : str, optional
            Label describing trace in caption, by default None
        """
        self.update_trace_data(self.n_history, x, y, label)

    def update_historic_data(self, x_values, y_values, labels=None):
        """Set historic XY traces

        Parameters
        ----------
        x_values : list[np.ndarray]
            List of x-values for each historic trace
        y_values : list[np.ndarray]
            List of y-values for each historic trace
        labels : list[str], optional
            List of labels for each historic trace, by default None
        """
        for n in range(self.n_history):
            # historic traces start with zero, which is the oldest
            trace_n = self.n_history - 1 - n
            if n < len(x_values) and n < len(x_values):
                label = None if labels is None else labels[n]
                self.update_trace_data(trace_n, x_values[n], y_values[n], label)
            else:
                # not enough data given, clear trace
                self.update_trace_data(trace_n, [], [], None)
