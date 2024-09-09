"""Widgets for plotting"""

import numpy as np
from matplotlib.image import PcolorImage
from qtpy import PYSIDE6, QtCharts, QtCore, QtGui
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor

from .models import Colormap

# this scale is used to circumwent a bug in QtCharts preventing data <1E-12
SCALE_FACTOR = float(2**32)  # 4294967296.0 ~ 10**9.63


class GraphItem:
    """Base class for graph items"""

    def __init__(self):
        super().__init__()
        self.min_x = np.nan
        self.max_x = np.nan
        self.min_y = np.nan
        self.max_y = np.nan
        self.min_z = np.nan
        self.max_z = np.nan
        # keep track of reference to graph, for propagating events to callbacks
        self.graph = None

    def _set_data_range(self, x: np.ndarray, y: np.ndarray, z: np.ndarray = None):
        """Store max/min data range for scaling purposes

        Parameters
        ----------
        x : np.ndarray
            X-values
        y : np.ndarray
            Y-values
        z : np.ndarray
            Z-values, by default None (no Z-values)
        """
        if len(y) > 0 and len(x) > 0:
            self.min_x = np.nanmin(x)
            self.max_x = np.nanmax(x)
            self.min_y = np.nanmin(y)
            self.max_y = np.nanmax(y)
        else:
            self.min_x = np.nan
            self.max_x = np.nan
            self.min_y = np.nan
            self.max_y = np.nan
        # add z values if given
        if z is None:
            return
        if len(z) > 0:
            self.min_z = np.nanmin(z)
            self.max_z = np.nanmax(z)
        else:
            self.min_z = np.nan
            self.max_z = np.nan

    def _on_pressed(self, point: QtCore.QPointF) -> None:
        """Callback for when a point is pressed

        Parameters
        ----------
        point : QtCore.QPointF
            Point that was clicked
        """
        # signals for mouse interactions are not precise, pass on to graph instead
        button = QtGui.QGuiApplication.mouseButtons()
        x_global = QtGui.QCursor.pos().x()
        y_global = QtGui.QCursor.pos().y()
        # map global position to graph widget position
        point_widget = self.graph.mapFromGlobal(QtCore.QPoint(x_global, y_global))
        x_pixel = point_widget.x()
        y_pixel = point_widget.y()
        # pass along to graph for handling mouse events
        self.graph.register_mouse_press(button, x_pixel, y_pixel)

    def _on_released(self, point: QtCore.QPointF) -> None:
        """Callback for when a mouse press is released

        Parameters
        ----------
        point : QtCore.QPointF
            Point that was clicked
        """
        # signals for mouse interactions are not precise, pass on to graph instead
        x_global = QtGui.QCursor.pos().x()
        y_global = QtGui.QCursor.pos().y()
        point_widget = self.graph.mapFromGlobal(QtCore.QPoint(x_global, y_global))
        x_pixel = point_widget.x()
        y_pixel = point_widget.y()
        # pass along to graph for handling mouse events
        self.graph.release_mouse_press(x_pixel, y_pixel)

    def _update_series(self, series: QtCharts.QXYSeries, x: np.ndarray, y: np.ndarray):
        """Update values for a QChart XY series, function depends on Qt wrapper type

        Parameters
        ----------
        series : QtCharts.QXYSeries
            Series to update
        x : np.ndarray
            X-data
        y : np.ndarray
            Y-data
        """
        # update function depends on Qt wrapper type
        if PYSIDE6 and isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            # pyside has a nicer interface with numpy arrays
            if SCALE_FACTOR == 1:
                # copy array to avoid issue with real/imag references
                series.replaceNp(np.copy(x), np.copy(y))
            else:
                # no need to copy when multiplying scale factor, will create new array
                series.replaceNp(SCALE_FACTOR * x, SCALE_FACTOR * y)
        else:
            values = [
                QtCore.QPointF(SCALE_FACTOR * xx, SCALE_FACTOR * yy)
                for xx, yy in zip(x, y)
            ]
            series.replace(values)

    def attach_to_graph(self, graph) -> None:
        """Attach graph item to graph

        Parameters
        ----------
        graph : Graph
            Graph to attach item to
        """
        # store graph reference, for propagating events to callbacks and later removal
        self.graph = graph

    def remove_from_graph(self) -> None:
        """Remove graph item from graph"""
        # remove graph reference
        self.graph = None


class XYItem(GraphItem):
    """Graph item representing a XY trace

    Parameters
    ----------
    x : list, optional
        x values, by default []
    y : list, optional
        y values, by default []
    color : str, optional
        Color description, by default '#000000'
    linewidth : int, optional
        linewidth, by default 1
    marker : bool, optional
        Whether to plot markers for each data point, by default True
    marker_size : int, optional
        Whether to plot markers for each data point, by default 4.5
    label : str, optional
        Label describing trace, by default ""
    """

    def __init__(
        self,
        x=[],
        y=[],
        color="#000000",
        linewidth=1,
        marker=True,
        marker_size=4.5,
        label: str = "",
    ):
        super().__init__()
        # create line series with data
        self.marker = marker
        self.marker_size = marker_size
        self.linewidth = linewidth
        self.color = color
        self.label = label
        self.series_line = QtCharts.QLineSeries()
        self.series_line.setPointsVisible(False)
        p = QtGui.QPen()
        p.setColor(QColor(color))
        p.setWidthF(linewidth)
        self.series_line.setPen(p)
        # set up callbacks
        self.series_line.pressed.connect(self._on_pressed)
        self.series_line.released.connect(self._on_released)
        # same for markers
        if marker:
            self.series_dot = QtCharts.QScatterSeries()
            self.series_dot.setColor(QColor(color))
            self.series_dot.setBorderColor(QColor(color))
            self.series_dot.setMarkerSize(self.marker_size)
            self.series_dot.pressed.connect(self._on_pressed)
            self.series_dot.released.connect(self._on_released)

        # update trace values
        self.update_values(x, y)

    def highlight_trace(self, highlight: bool = True) -> None:
        """Highlight trace

        Parameters
        ----------
        highlight : bool, optional
            Whether to highlight trace, by default True
        """
        if highlight:
            p = self.series_line.pen()
            p.setWidthF(3 * self.linewidth)
            self.series_line.setPen(p)
            if self.marker:
                self.series_dot.setMarkerSize(self.marker_size * 5 / 3)
        else:
            p = self.series_line.pen()
            p.setWidthF(self.linewidth)
            self.series_line.setPen(p)
            if self.marker:
                self.series_dot.setMarkerSize(self.marker_size)
        # call hide/show to force redraw
        self.series_line.hide()
        self.series_line.show()

    def attach_to_graph(self, graph) -> None:
        # overload of base function to add series to graph
        super().attach_to_graph(graph)
        # open gl uses non-scaled pixel-based linewidths, adjust accordingly
        self.series_line.setUseOpenGL(graph.use_open_gl)
        if graph.use_open_gl:
            self.linewidth *= 2.0
            self.marker_size *= 2.0
            p = self.series_line.pen()
            p.setWidthF(self.linewidth)
            self.series_line.setPen(p)
        # add series to graph
        self.graph.chart.addSeries(self.series_line)
        self.series_line.attachAxis(self.graph.x_axis_data)
        self.series_line.attachAxis(self.graph.y_axis_data)
        if self.marker:
            self.series_dot.setUseOpenGL(graph.use_open_gl)
            self.series_dot.setMarkerSize(self.marker_size)
            self.graph.chart.addSeries(self.series_dot)
            self.series_dot.attachAxis(self.graph.x_axis_data)
            self.series_dot.attachAxis(self.graph.y_axis_data)

    def remove_from_graph(self) -> None:
        # overload of base function to remove series from graph
        self.graph.chart.removeSeries(self.series_line)
        self.series_line.clear()
        if self.marker:
            self.graph.chart.removeSeries(self.series_dot)
            self.series_dot.clear()
        # remove graph reference at the end
        super().remove_from_graph()

    def update_values(self, x: np.ndarray, y: np.ndarray) -> None:
        """Update trace values

        Parameters
        ----------
        x : np.ndarray
            New x values
        y : np.ndarray
            New y values
        """
        # store data for selection and scaling purposes
        self._x = np.copy(x)
        self._y = np.copy(y)
        self._set_data_range(x, y)
        # update data points
        self._update_series(self.series_line, x, y)
        if self.marker:
            self._update_series(self.series_dot, x, y)


class MplImage(PcolorImage):
    """Extension of the matplotlib image class for rendering to a Qt dialog"""

    def render_image_qt(self, x0, x1, y0, y1, width, height):
        """Render image for use in Qt.

        This is a very minor modficiation of PColorImage.make_image.  The only
        change is the way the viewport is defined, from the axes.bbox object for
        the original function to input parameters for this function."""
        # background color should be transparent
        bg = np.array([0, 0, 0, 0], dtype=np.uint8)

        if self._imcache is None:
            A = self.to_rgba(self._A, bytes=True)
            # swap RGB to BGR
            A2 = np.array(A, dtype=np.uint8)
            A2[:, :, 0] = A[:, :, 2]
            A2[:, :, 2] = A[:, :, 0]
            self._imcache = np.pad(A2, [(1, 1), (1, 1), (0, 0)], "constant")
        padded_A = self._imcache

        if (padded_A[0, 0] != bg).all():
            padded_A[[0, -1], :] = padded_A[:, [0, -1]] = bg

        x_pix = np.linspace(x0, x1, width)
        y_pix = np.linspace(y1, y0, height)
        x_int = self._Ax.searchsorted(x_pix)
        y_int = self._Ay.searchsorted(y_pix)
        im = (  # See comment in NonUniformImage.make_image re: performance.
            padded_A.view(np.uint32)
            .ravel()[np.add.outer(y_int * padded_A.shape[1], x_int)]
            .view(np.uint8)
            .reshape((height, width, 4))
        )
        return im


class ImageItem(GraphItem):
    """Graph item representing a colormap image

    Parameters
    ----------
    x : np.ndarray
        Array with X values
    y : np.ndarray
        Array with Y values
    data : np.ndarray
        Image data, as 2D array
    cmap : Colormap
        Colormap to use
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, data: np.ndarray):
        super().__init__()
        # init intermediate representation
        self.mpl_image = None
        self._image_buffer = None
        self.pixmap = QtGui.QPixmap()
        self.x0 = None
        self.x1 = None
        self.y0 = None
        self.y1 = None
        # this is the matplotlib colormap to use - needs to be string to support _r
        self.mpl_cmap = "RdBu"
        # update data
        self.update_values(x, y, data)

    def update_values(self, x: np.ndarray, y: np.ndarray, data: np.ndarray) -> None:
        """Update image values

        Parameters
        ----------
        x : np.ndarray
            Array with X values
        y : np.ndarray
            Array with Y values
        data : np.ndarray
            Image data, as 2D array
        """
        # add edge values to x/y, centered around midpoint
        if len(x) == 1:
            x = np.array([x[0] * 0.95, x[0] * 1.05])
        elif len(x) > 1 and len(x) != (data.shape[1] + 1):
            x_mid = (x[1:] + x[:-1]) / 2
            x = np.concatenate(
                ([x_mid[0] - (x[1] - x[0])], x_mid, [x_mid[-1] + (x[-1] - x[-2])])
            )
        if len(y) == 1:
            y = np.array([y[0] * 0.95, y[0] * 1.05])
        elif len(y) > 1 and len(y) != (data.shape[0] + 1):
            y_mid = (y[1:] + y[:-1]) / 2
            y = np.concatenate(
                ([y_mid[0] - (y[1] - y[0])], y_mid, [y_mid[-1] + (y[-1] - y[-2])])
            )
        # check if x/y vectors are constant - if so, add some stretch to allow plotting
        if len(x) > 0 and np.all(x == x[0]):
            # scale deviation with fixed value unless it is zero
            dx = 1 if x[0] == 0 else x[0] * 1e-9
            x = np.linspace(x[0] - dx, x[0] + dx, len(x))
        # same for y
        if len(y) > 0 and np.all(y == y[0]):
            dy = 1 if y[0] == 0 else y[0] * 1e-9
            y = np.linspace(y[0] - dy, y[0] + dy, len(y))

        # init intermediate matplotlib representation
        if len(x) > 1 and len(y) > 1:
            self.mpl_image = MplImage(None, x, y, data, cmap=self.mpl_cmap, norm=None)
        else:
            self.mpl_image = None
        # store x/y data for scaling purposes
        self._set_data_range(x, y, data)

    def transpose(self) -> None:
        """Transpose image"""
        if self.mpl_image is None:
            return
        # create new image with transposed data
        x = self.mpl_image._Ay
        y = self.mpl_image._Ax
        data = self.mpl_image._A.T
        self.mpl_image = MplImage(None, x, y, data, cmap=self.mpl_cmap, norm=None)
        self._set_data_range(x, y, data)

    def set_colormap(self, colormap: Colormap, inverted=False) -> None:
        """Update colormap

        Parameters
        ----------
        colormap : Colormap
            Colormap to use
        inverted : bool, optional
            Whether to invert colormap, by default False
        """
        # create matplotlib colormap str
        mpl_cmap = colormap.name
        # red/blue, red/grey colormaps are inverted by default, reverse logic
        if mpl_cmap in ("RdBu", "RdGy"):
            if not inverted:
                mpl_cmap += "_r"
        else:
            if inverted:
                mpl_cmap += "_r"
        # store matplotlib colormap internally
        self.mpl_cmap = mpl_cmap
        if self.mpl_image is not None:
            self.mpl_image.set_cmap(mpl_cmap)

    def set_view(self, x0: int = None, x1: int = None, y0: int = None, y1: int = None):
        """Set viewport for image rendering, in data coordinates

        Parameters
        ----------
        x0 : int, optional
            Left limit of view, by default None
        x1 : int, optional
            Right view limit, by default None
        y0 : int, optional
            Lower view limit, by default None
        y1 : int, optional
            Upper view limit, by default None
        """
        if x0 is not None:
            self.x0 = x0
        if x1 is not None:
            self.x1 = x1
        if y0 is not None:
            self.y0 = y0
        if y1 is not None:
            self.y1 = y1

    def set_clim(self, z_min: float, z_max: float) -> None:
        """Set range of image data

        Parameters
        ----------
        z_min : float
            Minimum value
        z_max : float
            Maximum value
        """
        if self.mpl_image is not None:
            self.mpl_image.set_clim(z_min, z_max)

    def render_pixmap(self, width: int, height: int):
        """Render pixmap of image for use in qt dialog, using current view settings

        Parameters
        ----------
        width : int
            Width of pixmap
        height : int
            Height of pixmap
        """
        # return if view is not sets
        if self.x0 is None or self.x1 is None or self.y0 is None or self.y1 is None:
            return
        if self.mpl_image is None:
            return
        # render image using matplotlib
        self._image_buffer = self.mpl_image.render_image_qt(
            self.x0, self.x1, self.y0, self.y1, width, height
        )
        # convert to QImage, then pixmap
        self.qimage = QtGui.QImage(
            self._image_buffer.data,
            width,
            height,
            QtGui.QImage.Format.Format_ARGB32_Premultiplied,
        )
        self.pixmap.convertFromImage(
            self.qimage, Qt.ImageConversionFlag.NoFormatConversion
        )
