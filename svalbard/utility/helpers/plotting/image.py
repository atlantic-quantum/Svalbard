"""Widgets for plotting"""

import functools
import warnings

import numpy as np
from qtpy import QtCore, QtWidgets
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import QActionGroup, QMenu, QWidget
from superqt import QLabeledDoubleRangeSlider
from superqt.utils import signals_blocked

from .. import ui_tools
from .base_graph_ui import BaseGraphUI
from .graph_items import GraphItem, ImageItem
from .models import Colormap, ImageGraphModel, RangeType


class SliderAction(QtWidgets.QWidgetAction):
    """Widget action for range slider, for contrast toolbar item"""

    SliderUpdated = QtCore.Signal()
    AutoContrast = QtCore.Signal()

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        # keep track of slider range
        self._label: str = ""
        self._scaling = 1.0
        self._low: float = 0.0
        self._high: float = 1.0
        self._range_min: float = 0.0
        self._range_max: float = 1.0

    def createWidget(self, parent: QWidget) -> QWidget:
        """Create and format slider widget for to be used in action"""
        # add range slider for contrast
        slider = QLabeledDoubleRangeSlider(QtCore.Qt.Horizontal, parent=parent)
        slider.setMinimumWidth(400)
        slider.setRange(self._range_min, self._range_max)
        slider.setValue((self._low, self._high))
        slider.setDecimals(3)
        slider.setEdgeLabelMode(slider.EdgeLabelMode.LabelIsValue)
        slider.setHandleLabelPosition(slider.LabelPosition.NoLabel)
        slider.valueChanged.connect(self._on_slider)
        # combine wtih label and auto-contrast button
        label = QtWidgets.QLabel(self._label)
        button = QtWidgets.QToolButton()
        button.setText("Auto-contrast")
        button.clicked.connect(self._on_auto_contrast)
        layout = QtWidgets.QGridLayout()
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(2)
        layout.addWidget(label, 0, 0)
        layout.addWidget(button, 0, 1)
        layout.addWidget(slider, 1, 0, 1, 2)
        widget = QtWidgets.QWidget(parent=parent)
        widget.setLayout(layout)
        # store reference to silder, to be able to update
        widget.range_slider = slider
        widget.range_label = label
        return widget

    def set_label(self, label: str) -> None:
        """Set label of slider"""
        self._label = label
        for slider in self.createdWidgets():
            slider.range_label.setText(label)

    def get_range(self) -> tuple[float, float]:
        """Get range of slider"""
        return (self._low / self._scaling, self._high / self._scaling)

    def set_scaling(self, scaling: float) -> None:
        """Set scaling factor for slider parameters"""
        self._scaling = scaling

    def set_range(
        self, low: float, high: float, range_min: float = None, range_max: float = None
    ):
        """Set range of slider"""
        self._low = low * self._scaling
        self._high = high * self._scaling
        if range_min is not None:
            self._range_min = range_min * self._scaling
        if range_max is not None:
            self._range_max = range_max * self._scaling
        # update slider
        for slider in self.createdWidgets():
            with signals_blocked(slider.range_slider):
                slider.range_slider.setRange(self._range_min, self._range_max)
                slider.range_slider.setValue((self._low, self._high))

    def _on_slider(self, value: tuple[float, float]):
        """Callback on slider update"""
        # update slider values
        self._low = value[0]
        self._high = value[1]
        # emit signal for slider update
        self.SliderUpdated.emit()

    def _on_auto_contrast(self):
        """Callback on clicking the contrast button"""
        self.AutoContrast.emit()


class ImageGraph(BaseGraphUI):
    def __init__(self, use_open_gl=True, parent=None):
        """A graph that supports rendering of image colormaps

        Parameters
        ----------
        use_open_gl : bool, optional
            Whether to use openGL, by default True
        parent : QObject, optional
            Parent object, by default None
        """
        super().__init__(use_open_gl=use_open_gl, parent=parent)
        # image settings
        self.colormap = Colormap.RdBu
        self.invert_colormap = False
        self.transpose = False
        # margins for image
        self.x_margin = 0.006
        self.y_margin = 0.005
        # additional variables
        self.images = {}
        self.pixmap_item = None
        # make background black for image plots
        # self.chart.setPlotAreaBackgroundBrush(QColor("#222222"))
        # create image and colormap actions
        self._create_image_actions()
        # add contrast menu to toolbar, must be done after UI is created
        QtCore.QTimer.singleShot(0, self._add_contrast_action_menu)

    def _add_contrast_action_menu(self):
        """Add contrast menu to toolbar, must be done after UI is created"""
        self._menu_contrast.addAction(self.action_contrast_slider)

    def prepare_for_close(self):
        """Prepare for closing, make sure contrast menu is closed"""
        # close contrast menu, will cause leaks if open during quit
        self.action_contrast.menu().close()

    def _create_image_actions(self) -> None:
        """Create image actions for menu, toolbars and context menus"""
        # create actions for image manipulation
        self.action_transpose = ui_tools.create_action(
            self,
            "Transpose",
            self._on_transpose,
            tip="Transpose data",
            checkable=True,
            icon=ui_tools.get_theme_icon("transpose"),
        )
        self.action_contrast = ui_tools.create_action(
            self,
            "Contrast",
            self._on_contrast,
            tip="Enhance contrast",
            checkable=True,
            icon=ui_tools.get_theme_icon("contrast"),
        )
        # add range slider for contrast
        self._menu_contrast = QMenu()
        self.action_contrast.setMenu(self._menu_contrast)
        self.action_contrast_slider = SliderAction(self)
        self.action_contrast_slider.SliderUpdated.connect(self._on_contrast_slider)
        self.action_contrast_slider.AutoContrast.connect(self._on_auto_contrast)
        # add actions for colormap selection
        menu_color = QMenu()
        # add invert action
        self.action_invert_cmap = ui_tools.create_action(
            self,
            "Invert colormap",
            functools.partial(self._on_colormap, None),
            checkable=True,
        )
        menu_color.addAction(self.action_invert_cmap)
        menu_color.addSeparator()
        # create group with different colors
        color_group = QActionGroup(self)
        # mapping from colormap to action
        self._map_cmap_action = {}
        for colormap in Colormap:
            action = ui_tools.create_action(
                color_group,
                colormap.value,
                functools.partial(self._on_colormap, colormap),
                checkable=True,
                icon=f":/colormap-{colormap.name}.png",
            )
            # mark current colormap
            if colormap == self.colormap:
                action.setChecked(True)
            menu_color.addAction(action)
            self._map_cmap_action[colormap] = action
        # create main action that is owner of menu
        self.action_cmap = ui_tools.create_action(
            self,
            "Colormap",
            tip="Select colormap",
            icon=f":/colormap-{self.colormap.name}.png",
        )
        self.action_cmap.setMenu(menu_color)
        # update value of colormap actions
        self.set_colormap(self.colormap, self.invert_colormap)

    def _extra_context_actions(self) -> list[QtWidgets.QAction]:
        """Extra context menu actions for subclasses"""
        return [self.action_cmap, self.action_contrast, self.action_transpose, None]

    def get_toolbar(self) -> QtWidgets.QToolBar:
        # overload base class to include image actions
        toolbar = super().get_toolbar()
        # add actions
        toolbar.addSeparator()
        toolbar.addAction(self.action_cmap)
        toolbar.addAction(self.action_contrast)
        toolbar.addSeparator()
        toolbar.addAction(self.action_transpose)
        return toolbar

    def _get_all_graph_items(self) -> list[GraphItem]:
        # overload base class to include images
        parent_items = super()._get_all_graph_items()
        return parent_items + list(self.images.values())

    def _on_plot_area_changed(self, qrect: QtCore.QRect):
        """Callback function for plot area changes, will redraw the image

        Parameters
        ----------
        qrect : QtCore.QRect
            Current plot ares dimensions
        """
        super()._on_plot_area_changed(qrect)
        # force pixmap redraw
        if self.pixmap_item is None:
            return
        for image in self.images.values():
            image.render_pixmap(int(qrect.width()), int(qrect.height()))
            self.pixmap_item.setPixmap(image.pixmap)
            self.pixmap_item.setPos(qrect.x(), qrect.y())

    def update_ui_from_config(self) -> None:
        """Update UI from config"""
        super().update_ui_from_config()
        # update transpose action, but don't apply transpose (should be correct already)
        with signals_blocked(self.action_transpose):
            self.action_transpose.setChecked(self.transpose)
        # update colormaps actions
        self.set_colormap(self.colormap, self.invert_colormap)
        self.action_invert_cmap.setChecked(self.invert_colormap)
        self._map_cmap_action[self.colormap].setChecked(True)
        # update contrast action and image contrast
        self.action_contrast.setChecked(not self.autoscale_z)

    def update_config_from_ui(self) -> None:
        """Update config from UI"""
        super().update_config_from_ui()
        # update transpose and contrast
        self.autoscale_z = not self.action_contrast.isChecked()

    def _on_transpose(self):
        """Callback on transposing as image"""
        # do nothing if transpose state is not changed
        if self.transpose == self.action_transpose.isChecked():
            return
        self.transpose = self.action_transpose.isChecked()
        self._apply_transpose()
        # update UI and redraw image
        self.update_ui_from_config()
        self.redraw()

    def _apply_transpose(self) -> None:
        """Transpose axes and graph data"""
        # swap labels and units
        self.x_label, self.y_label = self.y_label, self.x_label
        self.x_unit, self.y_unit = self.y_unit, self.x_unit
        # swap axes ranges
        self.xlim, self.ylim = self.ylim, self.xlim
        self.autoscale_x, self.autoscale_y = self.autoscale_y, self.autoscale_x
        # transpose image data
        for image in self.images.values():
            image.transpose()
        # tranpose cursors
        if self.cursors is not None and self.cursors.style != RangeType.OFF:
            self.cursors.transpose()

    def calculate_auto_contrast(self):
        """Calculate auto-contrast range based on image data"""
        # get mpl image data
        im = self.images[0].mpl_image.get_array().ravel()
        # ignore warning related to mask
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # calculate range assuming some percenrtage of data is saturated
            clip = 40
            x1, x2 = np.nanpercentile(im, (clip / 2, 100 - clip / 2))
        self.set_zlim((x1, x2))

    def _on_contrast_slider(self):
        """Callback on contrast slider update"""
        # update contrast and redraw
        self.set_zlim(self.action_contrast_slider.get_range())
        self.redraw()

    def _on_auto_contrast(self):
        """Callback on clicking the contrast button"""
        self.calculate_auto_contrast()
        self.redraw()

    def _on_contrast(self):
        """Callback on contrast action update"""
        self.update_config_from_ui()
        # if contrast is enabled but limits not set, guess limits
        selection = self.action_contrast_slider.get_range()
        full_range = self._get_z_data_range()
        if not self.autoscale_z and np.allclose(selection, full_range, atol=1e-20):
            self.calculate_auto_contrast()
        # update contrast and redraw
        self.redraw()

    def _on_colormap(self, colormap=None):
        """Callback function for changing colormap"""
        # callback value could be none, if called from inverted action
        if colormap is None:
            colormap = self.colormap
        invert = self.action_invert_cmap.isChecked()
        self.set_colormap(colormap, invert)
        # redraw
        self.redraw()

    def set_colormap(self, colormap: Colormap, inverted=False):
        """Set colormap of image data

        Parameters
        ----------
        colormap : Colormap
            Colormap to use
        inverted : bool, optional
            Whether to invert colormap, by default False
        """
        self.colormap = colormap
        self.invert_colormap = inverted
        # update colormap
        for image in self.images.values():
            image.set_colormap(self.colormap, self.invert_colormap)
        # set action icon to current map
        self.action_cmap.setIcon(QIcon(f":/colormap-{self.colormap.name}.png"))

    def redraw(self) -> bool:
        # overload base class to include images
        super().redraw()
        qrect = self.chart.plotArea()
        # get either full range or user defined range
        zlim = self._get_z_data_range() if self.autoscale_z else self.zlim
        # set pixmap viewpoint and redraw
        for image in self.images.values():
            if len(zlim) == 2:
                image.set_clim(*zlim)
            image.set_view(self.xlim[0], self.xlim[1], self.ylim[0], self.ylim[1])
            image.render_pixmap(int(qrect.width()), int(qrect.height()))
            self.pixmap_item.setPixmap(image.pixmap)
            self.pixmap_item.setPos(qrect.x(), qrect.y())
        # update label of slider
        cfg = self._get_axes_config()
        self.action_contrast_slider.set_label(cfg["zlabel"])
        # update z slider range
        z_range = self._get_z_data_range()
        self.action_contrast_slider.set_scaling(self.z_scaling)
        self.action_contrast_slider.set_range(self.zlim[0], self.zlim[1], *z_range)
        return True

    def set_image(self, x: np.ndarray, y: np.ndarray, data: np.ndarray) -> None:
        """Create colormap image for graph

        Parameters
        ----------
        x : np.ndarray
            X-values
        y : np.ndarray
            Y-values
        data: np.ndarray
            Colormap values
        """
        # if transpose is enabled, swap x and y
        if self.transpose:
            x, y = y, x
            data = data.T
        # for now, only allow one image, but make compaitble with multiple
        image_n = 0
        # update values if image item already exists
        if image_n in self.images:
            image = self.images[image_n]
            image.update_values(x, y, data)
            return
        image = ImageItem(x, y, data)
        image.set_colormap(self.colormap, self.invert_colormap)
        # store image, add to scene
        self.images[image_n] = image
        self.pixmap_item = self.chart.scene().addPixmap(image.pixmap)
        # autoscale z, if wanted
        if self.autoscale_z:
            self.scale_z()

    def set_xlabel(self, label: str, unit: str = None) -> None:
        # overload base class to include transpose
        if self.transpose:
            super().set_ylabel(label, unit)
        else:
            super().set_xlabel(label, unit)

    def set_ylabel(self, label: str, unit: str = None) -> None:
        # overload base class to include transpose
        if self.transpose:
            super().set_xlabel(label, unit)
        else:
            super().set_ylabel(label, unit)

    def clear_all(self) -> None:
        # remove pixmap
        if self.pixmap_item is not None:
            self.chart.scene().removeItem(self.pixmap_item)
            self.pixmap_item = None
            self.images = {}
        # turn off transpose and contrast
        self.transpose = False
        self.invert_colormap = False
        # make sure transpose is unchecked, to avoid confusion between UI and config
        with signals_blocked(self.action_transpose):
            self.action_transpose.setChecked(False)
        super().clear_all()

    def _serialize(self) -> ImageGraphModel:
        """Serialize graph settings to pydantic model

        Returns
        -------
        ImageGraphModel
            Serialized graph settings
        """
        model_base = super()._serialize()
        model = ImageGraphModel(
            colormap=self.colormap,
            invert_colormap=self.invert_colormap,
            transpose=self.transpose,
            autoscale_z=self.autoscale_z,
            z_min=self.zlim[0],
            z_max=self.zlim[1],
            **model_base.model_dump(),
        )
        return model

    def _deserialize(self, model: ImageGraphModel) -> None:
        """Deserialize graph settings from pydantic model

        Parameters
        ----------
        model : ImageGraphModel
            Serialized graph settings
        """
        # important! do transpose first, since x/y axes may be swapped
        self.transpose = model.transpose
        if self.transpose != self.action_transpose.isChecked():
            self.action_transpose.setChecked(self.transpose)
            self._apply_transpose()
        # run parent class deserialization
        super()._deserialize(model)
        # update colormap if defined
        if model.colormap is not None:
            self.colormap = model.colormap
        self.invert_colormap = model.invert_colormap
        # contrast settings
        self.autoscale_z = model.autoscale_z
        self.zlim = np.array([model.z_min, model.z_max])
