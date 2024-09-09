"""Base widgets for plotting, including support for user interaction"""

import functools

import numpy as np
from qtpy import QtCore, QtGui, QtWidgets

from .. import ui_dialogs, ui_tools
from .base_graph import BaseGraph
from .cursors import RangeCursor
from .models import BaseGraphModel, ImageGraphModel, RangeType
from .ranges import SetAxesRangeDialog


class BaseGraphUI(BaseGraph):
    def __init__(self, use_open_gl=True, parent=None):
        """A base graph widget with user interaction (scrolling, zooming, etc.)

        The class is a subclass of BaseGraph, and adds user interaction.
        The class should be subclassed to add data (lines, images, etc).

        Parameters
        ----------
        use_open_gl : bool, optional
            Whether to use openGL, by default True
        parent : QObject, optional
            Parent object, by default None
        """
        super().__init__(use_open_gl, parent)
        # keep track of mouse events and interactions
        self._is_right_zooming = False
        self._is_panning = False
        self._is_mid_panning = False
        self._is_zoom_tool = False
        self._is_zoom_tool_x = False
        self._is_zoom_tool_y = False
        self._is_cursor_move = False
        self._is_mouse_moved = False
        self._mouse_press_coords = None
        self._key_press = ""
        self._org_xlim = np.array(self.xlim)
        self._org_ylim = np.array(self.ylim)
        self._rubber_band = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Rectangle, self)
        self._rubber_band.hide()
        # create user actions
        self._create_actions()
        # defer generating cursor until needed - will ensure cursor is on top of graph
        self.cursors = None

    def _on_plot_area_changed(self, qrect: QtCore.QRect):
        """Callback function for plot area changes, will redraw the image

        Parameters
        ----------
        qrect : QtCore.QRect
            Current plot ares dimensions
        """
        super()._on_plot_area_changed(qrect)
        if self.cursors is not None:
            self.cursors.update()

    def update_ui_from_config(self) -> None:
        """Update UI from config"""
        super().update_ui_from_config()
        self.action_scale_x.setChecked(self.autoscale_x)
        self.action_scale_y.setChecked(self.autoscale_y)
        self.action_log_x.setChecked(self.log_x)
        self.action_log_y.setChecked(self.log_y)
        # update cursor action icon
        if self.cursors is None or self.cursors.style == RangeType.OFF:
            # disable main cursor action but don't change icon
            self.action_cursor.setChecked(False)
        else:
            # called with cursor type, enable and change text and icon of main cursor
            self.action_cursor.setChecked(True)
            self.action_cursor.setText(self.cursors.style.value)
            self.action_cursor.setIcon(self._cursor_actions[self.cursors.style].icon())

    def update_config_from_ui(self) -> None:
        """Update config from UI"""
        super().update_config_from_ui()
        self.autoscale_x = self.action_scale_x.isChecked()
        self.autoscale_y = self.action_scale_y.isChecked()
        self.log_x = self.action_log_x.isChecked()
        self.log_y = self.action_log_y.isChecked()

    def _create_actions(self) -> None:
        """Create context menu actions for graph interactions"""
        self.action_select = ui_tools.create_action(
            self,
            "Select",
            tip="Select items in graph",
            checkable=True,
            icon=ui_tools.get_theme_icon("pointer"),
        )
        self.action_move = ui_tools.create_action(
            self,
            "Move/pan",
            tip="Move/pan graph",
            checkable=True,
            icon=ui_tools.get_theme_icon("move"),
        )
        self.action_zoom = ui_tools.create_action(
            self,
            "Zoom",
            tip="Zoom graph",
            checkable=True,
            icon=ui_tools.get_theme_icon("magnify"),
        )
        mouse_group = QtWidgets.QActionGroup(self)
        mouse_group.addAction(self.action_select)
        mouse_group.addAction(self.action_move)
        mouse_group.addAction(self.action_zoom)
        self.action_zoom_all = ui_tools.create_action(
            self,
            "Show all",
            self._on_autoscale,
            tip="Show all data in graph",
            checkable=False,
            icon=ui_tools.get_theme_icon("zoom-all"),
        )
        self.action_scale_x = ui_tools.create_action(
            self, "Autoscale X", self._on_scale_x, checkable=True
        )
        self.action_scale_x.setChecked(self.autoscale_x)
        self.action_scale_y = ui_tools.create_action(
            self, "Autoscale Y", self._on_scale_y, checkable=True
        )
        self.action_scale_y.setChecked(self.autoscale_y)
        self.action_log_x = ui_tools.create_action(
            self, "Log X", self._on_log_x_changed, checkable=True
        )
        self.action_log_x.setChecked(self.log_x)
        self.action_log_y = ui_tools.create_action(
            self, "Log Y", self._on_log_y_changed, checkable=True
        )
        self.action_log_y.setChecked(self.log_y)
        self.action_set_limits = ui_tools.create_action(
            self, "Set axes range...", self._on_set_axes_range
        )
        self.action_copy_image = ui_tools.create_action(
            self,
            "Copy image",
            self.copy_image_to_clipboard,
            tip="Copy graph to clipboard",
        )
        self.action_help = ui_tools.create_action(
            self, "Help...", self._on_help, tip="Show plot help"
        )
        # cursor actions
        menu_cursor = QtWidgets.QMenu()
        cursor_group = QtWidgets.QActionGroup(self)

        icons = {
            RangeType.OFF: None,
            RangeType.VERTICAL: ui_tools.get_theme_icon("cursor-vertical"),
            RangeType.HORIZONTAL: ui_tools.get_theme_icon("cursor-horizontal"),
            RangeType.BOTH: ui_tools.get_theme_icon("cursor"),
            RangeType.RANGE_VERTICAL: ui_tools.get_theme_icon("range-vertical"),
            RangeType.RANGE_HORIZONTAL: ui_tools.get_theme_icon("range-horizontal"),
            RangeType.RANGE: ui_tools.get_theme_icon("range"),
        }
        self._cursor_actions = {}
        for cursor_type in RangeType:
            action = ui_tools.create_action(
                cursor_group,
                cursor_type.value,
                functools.partial(self._on_cursor_selection, cursor_type),
                checkable=True,
                icon=icons[cursor_type],
            )
            menu_cursor.addAction(action)
            self._cursor_actions[cursor_type] = action
            # add separator after last item in each group
            if cursor_type in (RangeType.BOTH, RangeType.RANGE):
                menu_cursor.addSeparator()
            # mark cursor off action checked
            if cursor_type == RangeType.OFF:
                action.setChecked(True)
                menu_cursor.addSeparator()
        # combine to one menu item
        self.action_cursor = ui_tools.create_action(
            self,
            RangeType.BOTH.value,
            functools.partial(self._on_cursor_selection, None),
            checkable=True,
            icon=icons[RangeType.BOTH],
            tip="Select cursor type",
        )
        self.action_cursor.setMenu(menu_cursor)

    def get_toolbar(self) -> QtWidgets.QToolBar:
        """Get toolbar for graph, for use in main window

        Returns
        -------
        QtWidgets.QToolBar
            Toolbar with relevant tool actions
        """
        # create toolbar and set sizing
        toolbar = QtWidgets.QToolBar("Graph", self)
        toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        toolbar.setMovable(False)
        toolbar.setIconSize(QtCore.QSize(20, 20))
        # special zoom all button, with text
        tool_zoom_all = QtWidgets.QToolButton()
        tool_zoom_all.setDefaultAction(self.action_zoom_all)
        tool_zoom_all.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        # add actions
        toolbar.addAction(self.action_select)
        toolbar.addAction(self.action_move)
        toolbar.addAction(self.action_zoom)
        toolbar.addSeparator()
        toolbar.addWidget(tool_zoom_all)
        toolbar.addSeparator()
        toolbar.addAction(self.action_cursor)
        # the select action is selected by default
        self.action_select.setChecked(True)
        return toolbar

    def _on_cursor_selection(self, cursor_type: RangeType = None):
        """Callback for cursor interactions"""
        # generate cursor if not present
        if self.cursors is None:
            self.cursors = RangeCursor(graph=self)
        # if called without cursor type, trigger is main cursor action
        if cursor_type is None:
            # cursor type is stored in action text, set if enabled
            if self.action_cursor.isChecked():
                cursor_type = RangeType(self.action_cursor.text())
            else:
                cursor_type = RangeType.OFF
            # enable corresponding action in action group
            self._cursor_actions[cursor_type].setChecked(True)
        # update cursor and UI
        self.cursors.set_style(cursor_type)
        self.update_ui_from_config()

    def _on_autoscale(self) -> None:
        """Callback for scale all action"""
        # enable autoscale for both axes, then redraw to update UI
        self.autoscale_x = True
        self.autoscale_y = True
        self.update_ui_from_config()
        self.redraw()

    def _on_scale_x(self) -> None:
        """Callback for scale x action"""
        self.update_config_from_ui()
        if self.autoscale_x:
            # only redraw if autoscale is enabled
            self.redraw()

    def _on_scale_y(self) -> None:
        """Callback for scale y action"""
        self.update_config_from_ui()
        if self.autoscale_y:
            # only redraw if autoscale is enabled
            self.redraw()

    def _on_log_x_changed(self) -> None:
        """Callback for log x action"""
        pass

    def _on_log_y_changed(self) -> None:
        """Callback for log y action"""
        pass

    def _on_set_axes_range(self) -> None:
        """Callback axes range action"""
        # show dialog
        dlg = SetAxesRangeDialog(self)
        dlg.exec()

    def _on_help(self) -> None:
        """Show plot help dialog"""
        ui_dialogs.show_message_box(
            "Plot Help",
            """Plot functionality:

Left mouse button: Show item label
Left mouse button + space: Pan graph
Middle mouse button: Show all data
Middle mouse button + move: Pan graph
Right mouse button: Context menu
Right mouse button + move: Zoom graph

Left mouse + shift: Magnifier tool
Left mouse + 'x': Magnify horizontally
Left mouse + 'z': Magnify vertically

Scroll wheel: Zoom graph
Scroll wheel + 'x': Zoom horizontally
Scroll wheel + 'z': Zoom vertically
""",
            parent=self,
        )

    def keyPressEvent(self, event: QtGui.QKeyEvent):
        # Qt overload - store key press value for access in zoom events
        self._key_press = event.key()
        # abort interactions with escape
        if self._key_press == QtCore.Qt.Key_Escape:
            self.stop_user_interactions()

    def keyReleaseEvent(self, event: QtGui.QKeyEvent):
        # Qt overload - reset key press value
        self._key_press = None

    def wheelEvent(self, event: QtGui.QWheelEvent):
        # Qt overload - function for wheel event handling
        # ignore event if outside main plot area
        x = event.position().x()
        y = event.position().y()
        if self.is_point_outside_plot_area(x, y):
            event.ignore()
            return
        # calculate coordinates in plot area
        (xa, ya) = self.map_pixel_to_axes(x, y)
        # get step size - resolution may be finer for trackpads
        if event.hasPixelDelta():
            # for pixel scrolling, there is no minimum step
            step = event.pixelDelta().y() / 100
        else:
            # for angle, minimum step is 15 degrees = 120 points
            step = event.angleDelta().y() / 120
        # ignore events that don't actually scroll, e.g. trackpad tap
        if step == 0:
            event.ignore()
            return
        # make scrolling exponential in speed
        scale = 10 ** (-step / 10)
        # calculate and set new coordinates
        xlim = np.array(self.xlim)
        ylim = np.array(self.ylim)
        # scale x and y axes, with key modifiers
        if self._key_press != QtCore.Qt.Key_X:
            if self.log_y:
                yl = 10 ** (np.log10(ya) + scale * (np.log10(ylim) - np.log10(ya)))
                self.set_xlim(yl)
            else:
                self.set_ylim(ya + scale * (ylim - ya))
        if self._key_press != QtCore.Qt.Key_Z:
            if self.log_x:
                xl = 10 ** (np.log10(xa) + scale * (np.log10(xlim) - np.log10(xa)))
                self.set_xlim(xl)
            else:
                self.set_xlim(xa + scale * (xlim - xa))
        # rescale and redraw
        self.redraw()

    def mousePressEvent(self, event):
        # Qt overload - function for mouse press event handling
        # get coordinates of event
        xp = event.position().x()
        yp = event.position().y()
        # ignore event if outside main plot area
        if self.is_point_outside_plot_area(xp, yp):
            # show context menu outside plot area
            if event.button() == QtCore.Qt.MouseButton.RightButton:
                self._show_context_menu()
            event.ignore()
            return
        self.register_mouse_press(event.button(), xp, yp)
        event.accept()

    def register_mouse_press(self, button, xp: int, yp: int) -> bool:
        """Function for registering mouse press, for use in later graph manipulation

        Parameters
        ----------
        button : QtCore.Qt.MouseButton
            Mouse button pressed
        xp : int
            Mouse x pixel position, widget coordinates
        yp : int
            Mouse y pixel position, widget coordinates

        Returns
        -------
        bool
            Whether event was handled
        """
        # calculate coordinates in plot area
        (xa, ya) = self.map_pixel_to_axes(xp, yp)
        # store coordinate data and axes geometry to facilitate translations
        self._mouse_press_coords = [xp, yp, xa, ya]
        self._org_xlim = np.array(self.xlim)
        self._org_ylim = np.array(self.ylim)
        # check which button is pressed
        self._is_mouse_moved = False
        if button == QtCore.Qt.MouseButton.LeftButton:
            if self.action_move.isChecked():
                self._is_panning = True
            elif self.action_zoom.isChecked():
                self._is_zoom_tool = True
            elif self._key_press == QtCore.Qt.Key_Shift:
                self._is_zoom_tool = True
            elif self._key_press == QtCore.Qt.Key_X:
                self._is_zoom_tool_x = True
            elif self._key_press == QtCore.Qt.Key_Z:
                self._is_zoom_tool_y = True
            elif self._key_press == QtCore.Qt.Key_Space:
                self._is_panning = True
            elif self.cursors is not None and self.cursors.is_pressed(xp, yp):
                self._is_cursor_move = True
            else:
                # left button clicked without modifiers, no handled here
                return False
        elif button == QtCore.Qt.MouseButton.MiddleButton:
            self._is_mid_panning = True
        elif button == QtCore.Qt.MouseButton.RightButton:
            self._is_right_zooming = True
        # check for zoom tool
        if self._is_zoom_tool or self._is_zoom_tool_x or self._is_zoom_tool_y:
            # show rubber band at current mouse position
            self._rubber_band.setGeometry(xp, yp, 1, 1)
            self._rubber_band.show()
        # return True to mark event as handled
        return True

    def mouseDoubleClickEvent(self, event):
        # Qt overload - function for double-click handling
        # propegate double click event to mouse press event
        self.mousePressEvent(event)

    def mouseMoveEvent(self, event):
        # Qt overload - function for mouse move handling
        # ignore event if no mouse press event was registered
        if self._mouse_press_coords is None:
            event.ignore()
            return
        # mark that mouse has moved, for keeping track of click events
        self._is_mouse_moved = True
        xp, yp, xa, ya = self._mouse_press_coords
        # calculate shift in pixels compared to mouse press
        x = event.position().x()
        y = event.position().y()
        dx = x - xp
        dy = -(y - yp)
        width = self.chart.plotArea().width()
        height = self.chart.plotArea().height()
        # left button panning
        if self._is_panning or self._is_mid_panning:
            # calculate shift in axes coordinates
            dxa = dx * (self.xlim[1] - self.xlim[0]) / width
            dya = dy * (self.ylim[1] - self.ylim[0]) / height
            # rescale and redraw
            self.set_xlim(self._org_xlim - dxa)
            self.set_ylim(self._org_ylim - dya)
            self.redraw()
            event.accept()
            return
        # right button zooming
        if self._is_right_zooming:
            xscale = 10 ** (-1.25 * dx / width)
            yscale = 10 ** (-1.25 * dy / height)
            # calculate new limits, rescale and redraw
            self.set_xlim(xa + xscale * (self._org_xlim - xa))
            self.set_ylim(ya + yscale * (self._org_ylim - ya))
            self.redraw()
            event.accept()
            return
        if self._is_zoom_tool or self._is_zoom_tool_x or self._is_zoom_tool_y:
            # make sure rubber band is within plot area
            x1 = max(self.chart.plotArea().left(), min(x, xp))
            x2 = min(self.chart.plotArea().right(), max(x, xp))
            y1 = max(self.chart.plotArea().top(), min(y, yp))
            y2 = min(self.chart.plotArea().bottom(), max(y, yp))
            # check if vertical or horizontal zooming
            if self._is_zoom_tool_x:
                y1 = self.chart.plotArea().top()
                y2 = self.chart.plotArea().bottom()
            elif self._is_zoom_tool_y:
                x1 = self.chart.plotArea().left()
                x2 = self.chart.plotArea().right()
            self._rubber_band.setGeometry(x1, y1, x2 - x1, y2 - y1)
            event.accept()
            return
        if self._is_cursor_move:
            (xa_new, ya_new) = self.map_pixel_to_axes(x, y)
            self.cursors.drag(xa_new - xa, ya_new - ya)
        # ignore event if no action taken
        event.ignore()

    def mouseReleaseEvent(self, event):
        # Qt overload - function for mouse release event handling
        self.release_mouse_press(event.position().x(), event.position().y())
        event.accept()

    def release_mouse_press(self, x: int, y: int) -> None:
        """Function called after mouse release

        Parameters
        ----------
        x : int
            Mouse x pixel position in widget coordinates
        y : int
            Mouse y pixel position in widget coordinates
        """
        # zoom graph to new position
        if self._is_zoom_tool or self._is_zoom_tool_x or self._is_zoom_tool_y:
            xp, yp, xa, ya = self._mouse_press_coords
            xn, yn = self.map_pixel_to_axes(x, y)
            if not self._is_zoom_tool_y:
                self.set_xlim([min(xa, xn), max(xa, xn)])
            if not self._is_zoom_tool_x:
                self.set_ylim([min(ya, yn), max(ya, yn)])
            self.redraw()
        # show context menu for right-click and no mouse movement
        if self._is_right_zooming and not self._is_mouse_moved:
            self._show_context_menu()
        # zoom all for middle-click and no mouse movement
        if self._is_mid_panning and not self._is_mouse_moved:
            self._on_autoscale()
        # disable all actions
        self.stop_user_interactions()

    def stop_user_interactions(self) -> None:
        """Disable all mouse actions and hide tools"""
        # disable all mouse actions
        self._mouse_press_coords = None
        self._is_panning = False
        self._is_mid_panning = False
        self._is_right_zooming = False
        self._is_zoom_tool = False
        self._is_zoom_tool_x = False
        self._is_zoom_tool_y = False
        self._is_cursor_move = False
        if self.cursors is not None:
            self.cursors.release_drag()
        # self.cursor_2.release_drag()
        self._rubber_band.hide()

    def redraw(self) -> bool:
        # docstring inherited
        # call base clases and see if axes limits changed
        updated = super().redraw()
        if not updated:
            return False
        # update cursors to make length match new axis limits
        if self.cursors is not None:
            self.cursors.update()
        return True

    def _on_copy_cursor(self, cursor_value: float) -> None:
        """Callback function for copying cursor values to clipboard

        Parameters
        ----------
        cursor_value : float
            Value to be copied to clipboard
        """
        clipboard = QtWidgets.QApplication.clipboard()
        clipboard.setText(f"{cursor_value:.9g}")

    def _show_context_menu(self):
        """Display context menu for plot"""
        # create menu for copying cursor values
        copy_values_actions = []
        if self.cursors is not None and len(self.cursors._cursor_labels) > 0:
            self.action_copy = ui_tools.create_action(self, "Copy cursor values")
            menu_copy = QtWidgets.QMenu()
            for label, value in self.cursors._cursor_labels.items():
                # add separator if value is None
                if value is None:
                    menu_copy.addSeparator()
                    continue
                action = ui_tools.create_action(
                    menu_copy,
                    label,
                    functools.partial(self._on_copy_cursor, value),
                    tip="Copy cursor value to clipboard",
                )
                menu_copy.addAction(action)
            self.action_copy.setMenu(menu_copy)
            copy_values_actions = [self.action_copy]
        # context menu options
        context_actions = (
            [self.action_zoom_all, None]
            + self._extra_context_actions()
            + [self.action_cursor]
            + copy_values_actions
            + [
                None,
                self.action_copy_image,
                None,
                self.action_set_limits,
                None,
                self.action_scale_x,
                self.action_scale_y,
                None,
                self.action_log_x,
                self.action_log_y,
                None,
                self.action_help,
            ]
        )
        # create main context menu
        menu = QtWidgets.QMenu(self)
        ui_tools.add_actions(menu, context_actions)
        # show menu at mouse cursor position
        menu.popup(QtGui.QCursor.pos())

    def _extra_context_actions(self) -> list[QtWidgets.QAction]:
        """Extra context menu actions for subclasses

        This function should be subclassed.

        Returns
        -------
        list[QtWidgets.QAction]
            Actions to be added to context menu.
        """
        return []

    def clear_all(self) -> None:
        # clear cursors, then remove
        if self.cursors is not None:
            self.cursors.set_style(RangeType.OFF)
        self.cursors = None
        # mark cursor off action checked and main cursor action off
        action_cursor_off = self._cursor_actions[RangeType.OFF]
        action_cursor_off.setChecked(True)
        self.action_cursor.setChecked(False)
        # call base class to remove rest
        super().clear_all()

    def _serialize(self) -> BaseGraphModel:
        """Serialize graph settings to pydantic model

        Returns
        -------
        BaseGraphModel
            Serialized graph settings
        """
        # make sure cursor exists
        if self.cursors is None:
            self.cursors = RangeCursor(graph=self)
        model = BaseGraphModel(
            log_x=self.log_x,
            log_y=self.log_y,
            x_min=self.xlim[0],
            x_max=self.xlim[1],
            y_min=self.ylim[0],
            y_max=self.ylim[1],
            autoscale_x=self.autoscale_x,
            autoscale_y=self.autoscale_y,
            cursors=self.cursors.serialize(),
        )
        return model

    def _deserialize(self, model: BaseGraphModel) -> None:
        """Deserialize graph settings from pydantic model

        Parameters
        ----------
        model : BaseGraphModel
            Serialized graph settings
        """
        self.log_x = model.log_x
        self.log_y = model.log_y
        self.xlim = np.array([model.x_min, model.x_max])
        self.ylim = np.array([model.y_min, model.y_max])
        self.autoscale_x = model.autoscale_x
        self.autoscale_y = model.autoscale_y
        # make sure cursor exists
        if self.cursors is None:
            self.cursors = RangeCursor(graph=self)
        self.cursors.deserialize(model.cursors)

    def dump_to_model(self) -> ImageGraphModel:
        """Dump graph settings to pydantic model

        Returns
        -------
        ImageGraphModel
            Serialized graph settings
        """
        # first, make sure config from UI is up to date
        self.update_config_from_ui()
        # use serialize function to get model, will call base classes
        return self._serialize()

    def load_from_model(self, model: ImageGraphModel) -> None:
        """Load graph settings from pydantic model

        Parameters
        ----------
        model : ImageGraphModel
            Serialized graph settings
        """
        # use deserialize function to set model, will call base classes
        self._deserialize(model)
        # update UI elements and redraw
        self.update_ui_from_config()
        self.redraw()
