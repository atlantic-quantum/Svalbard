"""Widgets for plotting"""

from qtpy import QtCharts, QtCore, QtGui, QtWidgets
from qtpy.QtGui import QColor

from ... import str_helper
from .. import ui_tools
from .graph_items import GraphItem
from .models import CursorModel, CursorType, RangeType


class SingleCursor(GraphItem):
    """Graph item representing a single cursor."""

    # cursor linewidth
    LINEWIDTH = 2

    def __init__(self):
        super().__init__()
        self.linewidth = self.LINEWIDTH
        self.color = QColor("#DDDDDD") if ui_tools.is_dark_mode() else QColor("#000000")
        self.style = CursorType.OFF
        self.x = 0.0
        self.y = 0.0
        # keep track of mouse press/drag
        self.x_pressed = None
        self.y_pressed = None
        # create line series with data
        self.series_h = QtCharts.QLineSeries()
        self.series_v = QtCharts.QLineSeries()
        self.series_h.setPointsVisible(False)
        self.series_v.setPointsVisible(False)
        p = QtGui.QPen()
        p.setWidthF(self.linewidth)
        p.setColor(QColor(self.color))
        self.series_h.setPen(p)
        self.series_v.setPen(p)
        # set up callbacks
        self.series_h.pressed.connect(self._on_pressed)
        self.series_h.released.connect(self._on_released)
        self.series_v.pressed.connect(self._on_pressed)
        self.series_v.released.connect(self._on_released)

    def attach_to_graph(self, graph) -> None:
        # overload of base function to add series to graph
        super().attach_to_graph(graph)
        # add series to graph
        self.series_h.setUseOpenGL(graph.use_open_gl)
        self.series_v.setUseOpenGL(graph.use_open_gl)
        # open gl uses non-scaled pixel-based linewidths, adjust accordingly
        if graph.use_open_gl:
            self.linewidth = 2.0 * self.LINEWIDTH
            p = self.series_h.pen()
            p.setWidthF(self.linewidth)
            self.series_h.setPen(p)
            self.series_v.setPen(p)
        self.graph.chart.addSeries(self.series_h)
        self.series_h.attachAxis(self.graph.x_axis_data)
        self.series_h.attachAxis(self.graph.y_axis_data)
        self.graph.chart.addSeries(self.series_v)
        self.series_v.attachAxis(self.graph.x_axis_data)
        self.series_v.attachAxis(self.graph.y_axis_data)

    def remove_from_graph(self) -> None:
        # overload of base function to remove series from graph
        self.graph.chart.removeSeries(self.series_h)
        self.series_h.clear()
        self.graph.chart.removeSeries(self.series_v)
        self.series_v.clear()
        # remove graph reference at the end
        # super().remove_from_graph()

    def is_pressed(self, x: int, y: int) -> bool:
        """Check if cursor is near given graph widget pixel coordinates

        Parameters
        ----------
        x : int
            X position, graph widget pixel coordinates
        y : int
            Y position, graph widget pixel coordinates

        Returns
        -------
        bool
            True if cursor is pressed
        """
        # compare position in pixel units - relevant for mouse precision
        (xp, yp) = self.graph.map_axes_to_pixel(self.x, self.y)
        if self.style in (CursorType.VERTICAL, CursorType.BOTH):
            if abs(xp - x) <= 4:
                self.mark_as_pressed(x=True)
        if self.style in (CursorType.HORIZONTAL, CursorType.BOTH):
            if abs(yp - y) <= 4:
                self.mark_as_pressed(y=True)
        # update cursor if pressed to reflect new line styles
        pressed = self.x_pressed is not None or self.y_pressed is not None
        if pressed:
            self.update()
        return pressed

    def mark_as_pressed(self, x: bool = None, y: bool = None):
        """Set cursor pressed state for x, y, or both axes

        Parameters
        ----------
        x : bool, optional
            Mark x as pressed, by default None (leave unchanged)
        y : bool, optional
            Mark y as pressed, by default None (leave unchanged)
        """
        if x is not None:
            self.x_pressed = self.x if x else None
        if y is not None:
            self.y_pressed = self.y if y else None
        # highlight pressed cursor line
        if self.x_pressed is not None:
            p = self.series_v.pen()
            p.setWidthF(1.5 * self.linewidth)
            p.setColor(QColor("blue"))
            self.series_v.setPen(p)
        if self.y_pressed is not None:
            p = self.series_h.pen()
            p.setWidthF(1.5 * self.linewidth)
            p.setColor(QColor("blue"))
            self.series_h.setPen(p)

    def drag(self, dx: float, dy: float):
        """Drag cursor to new position.  Only previously pressed items will be updated

        Parameters
        ----------
        dx : float
            Change in x position since mouse press, in axis units
        dy : float
            Change in y position since mouse press, in axis units
        """
        # only drag line if coordinate was previously pressed
        if self.x_pressed is not None:
            self.x = self.x_pressed + dx
        if self.y_pressed is not None:
            self.y = self.y_pressed + dy

    def release_drag(self):
        """Release cursor from drag mode."""
        self.mark_as_pressed(x=False, y=False)
        p = self.series_v.pen()
        p.setWidthF(self.linewidth)
        p.setColor(QColor(self.color))
        self.series_h.setPen(p)
        self.series_v.setPen(p)
        self.update()

    def update(self, x: float = None, y: float = None) -> None:
        """Set cursor position and update plot values according to axis range.

        Parameters
        ----------
        x : float, optional
            x position
        y : float, optional
            y position
        """
        if x is not None:
            self.x = x
        if y is not None:
            self.y = y
        # if item is attached to graph, update cursor length to match graph limits
        if self.graph is None:
            return
        if self.style == CursorType.VERTICAL:
            self._update_series(self.series_h, [], [])
            self._update_series(self.series_v, [self.x, self.x], self.graph.ylim)
        elif self.style == CursorType.HORIZONTAL:
            self._update_series(self.series_h, self.graph.xlim, [self.y, self.y])
            self._update_series(self.series_v, [], [])
        elif self.style == CursorType.BOTH:
            self._update_series(self.series_h, self.graph.xlim, [self.y, self.y])
            self._update_series(self.series_v, [self.x, self.x], self.graph.ylim)
        else:
            self._update_series(self.series_h, [], [])
            self._update_series(self.series_v, [], [])

    def set_style(self, style: CursorType, reset_position=0.5) -> None:
        """Set cursor style and update cursor position according to axis range.

        Parameters
        ----------
        style : CursorType
            Cursor line style
        reset_position : float
            Cursor reset position, scaled to range. Applied if cursor is outside axes
        """
        # if previous stye was off, always reset position
        perform_reset = self.style == CursorType.OFF
        self.style = style
        # if item is attached to graph, update cursor length to match graph limits
        if self.graph is None:
            return
        # position cursor to left/right up/down pending reset position
        xlim = self.graph.xlim
        ylim = self.graph.ylim
        if perform_reset or self.x < xlim[0] or self.x > xlim[1]:
            self.x = xlim[0] + reset_position * (xlim[1] - xlim[0])
        if perform_reset or self.y < ylim[0] or self.y > ylim[1]:
            self.y = ylim[0] + reset_position * (ylim[1] - ylim[0])
        # call general update to redraw
        self.update()

    def transpose(self) -> None:
        """Transpose cursor type and position without redrawing."""
        self.x, self.y = self.y, self.x
        if self.style == CursorType.VERTICAL:
            self.style = CursorType.HORIZONTAL
        elif self.style == CursorType.HORIZONTAL:
            self.style = CursorType.VERTICAL


class RangeCursor(GraphItem):
    """Graph item representing a point or a range, using one or two line cursors

    Parameters
    ----------
    graph : Graph
        Graph cursor belongs to
    """

    def __init__(self, graph):
        super().__init__()
        self.style = RangeType.OFF
        self.cursor1 = SingleCursor()
        self.cursor2 = SingleCursor()
        # store active cursor labels, to be used in copy menu
        self._cursor_labels = {}
        # create cursor shade area
        self.shade_area = QtWidgets.QGraphicsRectItem()
        self.shade_area.setPen(QtCore.Qt.NoPen)
        self.shade_area.setBrush(
            QtGui.QColor("#4A4A4A" if ui_tools.is_dark_mode() else "#ABC9F5")
        )
        # make cursor shade transparent and sit on top of graph
        self.shade_area.setOpacity(0.3)
        self.shade_area.setZValue(100)
        # create cursor labels
        f = QtGui.QFont("Arial", 14 if ui_tools.MAC else 10)
        self.label_left = QtWidgets.QGraphicsSimpleTextItem("")
        self.label_center = QtWidgets.QGraphicsSimpleTextItem("")
        self.label_right = QtWidgets.QGraphicsSimpleTextItem("")
        self.label_left.setFont(f)
        self.label_center.setFont(f)
        self.label_right.setFont(f)
        # dark mode style changes
        if ui_tools.is_dark_mode():
            self.label_left.setBrush(QtGui.QColor("#DDDDDD"))
            self.label_center.setBrush(QtGui.QColor("#DDDDDD"))
            self.label_right.setBrush(QtGui.QColor("#DDDDDD"))
        # attach to graph at init
        self.attach_to_graph(graph)

    def attach_to_graph(self, graph) -> None:
        # overload of base function to add series to graph
        super().attach_to_graph(graph)
        # add two cursors to graph
        self.cursor1.attach_to_graph(graph)
        self.cursor2.attach_to_graph(graph)
        # add area patch
        self.graph.chart.scene().addItem(self.shade_area)
        self.graph.chart.scene().addItem(self.label_left)
        self.graph.chart.scene().addItem(self.label_right)
        self.graph.chart.scene().addItem(self.label_center)

    def remove_from_graph(self) -> None:
        # overload of base function to remove series from graph
        self.cursor1.remove_from_graph()
        self.cursor2.remove_from_graph()
        # remove area patch
        self.graph.chart.scene().removeItem(self.shade_area)
        self.graph.chart.scene().removeItem(self.label_left)
        self.graph.chart.scene().removeItem(self.label_right)
        self.graph.chart.scene().removeItem(self.label_center)
        # remove graph reference at the end
        super().remove_from_graph()

    def is_pressed(self, x: int, y: int) -> bool:
        """Check if cursors are near given graph widget pixel coordinates

        Parameters
        ----------
        x : int
            X position, graph widget pixel coordinates
        y : int
            Y position, graph widget pixel coordinates

        Returns
        -------
        bool
            True if cursor is pressed
        """
        # only allow one cursor to be pressed, only check cursor 2 if 1 is not
        pressed = self.cursor1.is_pressed(x, y)
        if not pressed:
            pressed = self.cursor2.is_pressed(x, y)
        # only check for press of area cursor if lines are not pressed
        if not pressed:
            # get range for area between cursors
            x1 = min(self.cursor1.x, self.cursor2.x)
            x2 = max(self.cursor1.x, self.cursor2.x)
            y1 = min(self.cursor1.y, self.cursor2.y)
            y2 = max(self.cursor1.y, self.cursor2.y)
            (xa, ya) = self.graph.map_pixel_to_axes(x, y)
            # check for user interaction between cursors
            if self.style == RangeType.RANGE_HORIZONTAL:
                pressed = y1 <= ya <= y2
                if pressed:
                    self.cursor1.mark_as_pressed(y=True)
                    self.cursor2.mark_as_pressed(y=True)
            if self.style == RangeType.RANGE_VERTICAL:
                pressed = x1 <= xa <= x2
                if pressed:
                    self.cursor1.mark_as_pressed(x=True)
                    self.cursor2.mark_as_pressed(x=True)
            if self.style == RangeType.RANGE:
                pressed = (x1 <= xa <= x2) and (y1 <= ya <= y2)
                if pressed:
                    self.cursor1.mark_as_pressed(x=True, y=True)
                    self.cursor2.mark_as_pressed(x=True, y=True)
            # redraw cursor if area patch was pressed
            if pressed:
                self.update()
        return pressed

    def drag(self, dx: float, dy: float):
        """Drag cursor to new position.  Only previously pressed items will be updated

        Parameters
        ----------
        dx : float
            Change in x position since mouse press, in axis units
        dy : float
            Change in y position since mouse press, in axis units
        """
        # update both cursors
        self.cursor1.drag(dx, dy)
        self.cursor2.drag(dx, dy)
        self.update()

    def release_drag(self):
        """Release cursors from drag mode."""
        self.cursor1.release_drag()
        self.cursor2.release_drag()

    def set_style(self, style: RangeType) -> None:
        """Set style of cursors.

        Parameters
        ----------
        style : RangeType
            Cursor style
        """
        self.style = style
        # get mapping for the individual cursor line styles
        CURSOR_TYPES = {
            RangeType.VERTICAL: (CursorType.VERTICAL, CursorType.OFF),
            RangeType.HORIZONTAL: (CursorType.HORIZONTAL, CursorType.OFF),
            RangeType.BOTH: (CursorType.BOTH, CursorType.OFF),
            RangeType.RANGE_VERTICAL: (CursorType.VERTICAL, CursorType.VERTICAL),
            RangeType.RANGE_HORIZONTAL: (CursorType.HORIZONTAL, CursorType.HORIZONTAL),
            RangeType.RANGE: (CursorType.BOTH, CursorType.BOTH),
            RangeType.OFF: (CursorType.OFF, CursorType.OFF),
        }
        cursor_1_type, cursor_2_type = CURSOR_TYPES[style]
        self.cursor1.set_style(cursor_1_type, reset_position=0.3)
        self.cursor2.set_style(cursor_2_type, reset_position=0.7)
        self.update()

    def update(self) -> None:
        """Update cursor to reflect coordinates and axis range."""
        # update single cursors
        self.cursor1.update()
        self.cursor2.update()
        if self.graph is None:
            return
        # get cursor values for labels and shaded area
        xlim = self.graph.xlim
        ylim = self.graph.ylim
        x1 = min(self.cursor1.x, self.cursor2.x)
        x2 = max(self.cursor1.x, self.cursor2.x)
        y1 = min(self.cursor1.y, self.cursor2.y)
        y2 = max(self.cursor1.y, self.cursor2.y)

        # update/show/hide shaded aread depending on range-like or single cursor
        RT = RangeType
        if self.style in (RT.OFF, RT.VERTICAL, RT.HORIZONTAL, RT.BOTH):
            # for single cursor, ignore sorting and always use cursor 1 value
            x1 = self.cursor1.x
            y1 = self.cursor1.y
            # hide shaded area for single cursors
            self.shade_area.setVisible(False)
        else:
            # get axes coordinates of shaded area between cursors, including axes limits
            if self.style == RangeType.RANGE_VERTICAL:
                ax = [max(x1, xlim[0]), min(x2, xlim[1])]
                ay = ylim
            elif self.style == RangeType.RANGE_HORIZONTAL:
                ax = xlim
                ay = [max(y1, ylim[0]), min(y2, ylim[1])]
            elif self.style == RangeType.RANGE:
                ax = [max(x1, xlim[0]), min(x2, xlim[1])]
                ay = [max(y1, ylim[0]), min(y2, ylim[1])]
            # hide shaded area if outside axes
            if ax[0] > xlim[1] or ax[1] < xlim[0] or ay[0] > ylim[1] or ay[1] < ylim[0]:
                self.shade_area.setVisible(False)
            else:
                # update shape and show shaded area
                px1, py1 = self.graph.map_axes_to_pixel(ax[0], ay[0])
                px2, py2 = self.graph.map_axes_to_pixel(ax[1], ay[1])
                self.shade_area.setRect(px1, py1, px2 - px1, py2 - py1)
                self.shade_area.setVisible(True)

        # update cursor labels
        self._update_labels(x1, y1, x2, y2)

    def transpose(self) -> None:
        """Transpose cursor type and position"""
        self.cursor1.transpose()
        self.cursor2.transpose()
        if self.style == RangeType.VERTICAL:
            self.style = RangeType.HORIZONTAL
        elif self.style == RangeType.HORIZONTAL:
            self.style = RangeType.VERTICAL
        elif self.style == RangeType.RANGE_VERTICAL:
            self.style = RangeType.RANGE_HORIZONTAL
        elif self.style == RangeType.RANGE_HORIZONTAL:
            self.style = RangeType.RANGE_VERTICAL
        # final call to update will update all subcomponents
        self.update()

    def _update_labels(self, x1: int, y1: int, x2: int, y2: int) -> None:
        """Update cursor labels

        Parameters
        ----------
        x1 : int
            Cursor 1 x position
        y1 : int
            Cursor 1 y position
        x2 : int
            Cursor 2 x position
        y2 : int
            Cursor 2 y position
        """
        # create cursor labels
        x1_text = str_helper.get_si_string(x1, self.graph.x_unit, 5, True)
        x2_text = str_helper.get_si_string(x2, self.graph.x_unit, 5, True)
        y1_text = str_helper.get_si_string(y1, self.graph.y_unit, 5, True)
        y2_text = str_helper.get_si_string(y2, self.graph.y_unit, 5, True)
        xm_text = str_helper.get_si_string((x1 + x2) / 2, self.graph.x_unit, 5, True)
        xr_text = str_helper.get_si_string(x2 - x1, self.graph.x_unit, 5, True)
        ym_text = str_helper.get_si_string((y1 + y2) / 2, self.graph.y_unit, 5, True)
        yr_text = str_helper.get_si_string(y2 - y1, self.graph.y_unit, 5, True)
        # store cursor labels and values for copy menu
        self._cursor_labels = {}
        if self.style == RangeType.VERTICAL:
            text_l = f"x: {x1_text}"
            text_c = ""
            text_r = ""
            self._cursor_labels[f"x: {x1_text}"] = x1
        elif self.style == RangeType.HORIZONTAL:
            text_l = f"y: {y1_text}"
            text_c = ""
            text_r = ""
            self._cursor_labels[f"y: {y1_text}"] = y1
        elif self.style == RangeType.BOTH:
            text_l = f"x: {x1_text}, y: {y1_text}"
            text_c = ""
            text_r = ""
            self._cursor_labels[f"x: {x1_text}"] = x1
            self._cursor_labels[f"y: {y1_text}"] = y1
        elif self.style == RangeType.RANGE_VERTICAL:
            text_l = f"x1: {x1_text}"
            text_c = f"<x>: {xm_text},  dx: {xr_text}"
            text_r = f"x2: {x2_text}"
            self._cursor_labels[f"x1: {x1_text}"] = x1
            self._cursor_labels[f"x2: {x2_text}"] = x2
            self._cursor_labels[f"<x>: {xm_text}"] = (x1 + x2) / 2
            self._cursor_labels[f"dx: {xr_text}"] = x2 - x1
        elif self.style == RangeType.RANGE_HORIZONTAL:
            text_l = f"y1: {y1_text}"
            text_c = f"<y>: {ym_text},  dy: {yr_text}"
            text_r = f"y2: {y2_text}"
            self._cursor_labels[f"y1: {y1_text}"] = y1
            self._cursor_labels[f"y2: {y2_text}"] = y2
            self._cursor_labels[f"<y>: {ym_text}"] = (y1 + y2) / 2
            self._cursor_labels[f"dy: {yr_text}"] = y2 - y1
        elif self.style == RangeType.RANGE:
            text_l = f"x1: {x1_text}, y1: {y1_text}"
            text_c = ""
            text_r = f"x2: {x2_text}, y2: {y2_text}"
            self._cursor_labels[f"x1: {x1_text}"] = x1
            self._cursor_labels[f"x2: {x2_text}"] = x2
            self._cursor_labels[f"<x>: {xm_text}"] = (x1 + x2) / 2
            self._cursor_labels[f"dx: {xr_text}"] = x2 - x1
            self._cursor_labels["Separator"] = None
            self._cursor_labels[f"y1: {y1_text}"] = y1
            self._cursor_labels[f"y2: {y2_text}"] = y2
            self._cursor_labels[f"<y>: {ym_text}"] = (y1 + y2) / 2
            self._cursor_labels[f"dy: {yr_text}"] = y2 - y1
        else:
            text_l = ""
            text_c = ""
            text_r = ""
        # update cursor labels
        self.label_left.setText(text_l)
        self.label_center.setText(text_c)
        self.label_right.setText(text_r)

        # update cursor label position
        left = self.graph.chart.plotArea().left()
        width = self.graph.chart.plotArea().width()
        right = left + width - self.label_right.boundingRect().width()
        center = left + width / 2 - self.label_center.boundingRect().width() / 2
        top = self.graph.chart.plotArea().top() - 20
        # for single cursor, make label follow x position
        if self.style in (RangeType.VERTICAL, RangeType.BOTH):
            px1, py1 = self.graph.map_axes_to_pixel(x1, y1)
            cursor_x = px1 - self.label_left.boundingRect().width() / 2
            max_x = left + width - self.label_left.boundingRect().width()
            left = min(max(left, cursor_x), max_x)
        # update positions
        self.label_left.setPos(left, top)
        self.label_center.setPos(center, top)
        self.label_right.setPos(right, top)

    def serialize(self) -> CursorModel:
        """Serialize cursor to pydantic model

        Returns
        -------
        CursorModel
            Pydantic model with cursor data
        """
        model = CursorModel(
            style=self.style,
            x1=self.cursor1.x,
            x2=self.cursor2.x,
            y1=self.cursor1.y,
            y2=self.cursor2.y,
        )
        return model

    def deserialize(self, model: CursorModel) -> None:
        """Deserialize cursor from pydantic model

        Parameters
        ----------
        model : CursorModel
            Pydantic model with cursor data
        """
        self.set_style(model.style)
        self.cursor1.update(x=model.x1, y=model.y1)
        self.cursor2.update(x=model.x2, y=model.y2)
