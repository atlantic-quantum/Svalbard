"""Qt controls for each of the allowed parameter datatypes."""

import functools
from enum import Enum
from typing import Any

from qtpy import QtCore, QtWidgets
from qtpy.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QSizePolicy,
    QWidget,
)

from . import ui_widgets


def create_control(data: Any, parent: QWidget = None) -> QWidget:
    """Create control for data type.

    Parameters
    ----------
    data : Any
        Data to create control for
    parent : QWidget
        Parent QWidget owning control

    Returns
    -------
    QWidget
        Control for data type
    """
    # create control based on data type
    if isinstance(data, bool):
        return BoolControl(parent)
    if isinstance(data, complex):
        return ComplexControl(parent)
    if isinstance(data, float):
        return FloatControl(parent)
    if isinstance(data, int):
        return IntControl(parent)
    if isinstance(data, str):
        return StringControl(parent)
    if isinstance(data, Enum):
        return EnumControl(type(data), parent)
    if isinstance(data, dict):
        return DictControl(data, parent)
    if isinstance(data, list):
        return ListControl(data, parent)
    else:
        raise TypeError(f"Cannot create control for type {type(data)}")


# can unfortunatly not be defined as a metaclass b/c issue with metaclasses from PyQt
class WidgetInterface:
    """Interface for unifying interactions with UI elements"""

    # define a few functions that needs to be implemented by subclasses
    def get_value(self):
        """Get value of UI element.

        Returns
        -------
        Value
            Current value of UI element. Type is given by the control type.
        """
        raise NotImplementedError

    def set_value(self, value, block_signals=True) -> None:
        """Set value of UI element.

        Parameters
        ----------
        value
            New value
        block_signals : bool
            Whether to block signals when setting value
        """
        raise NotImplementedError

    def set_callback(self, callback: callable, with_value=True) -> None:
        """Set callback function to be called when UI element is changed.

        Parameters
        ----------
        callback : callable
            Function to call when UI element is changed
        with_value : bool
            If True, the callback will be called with the new value as parameter
        """
        if with_value:
            callback = functools.partial(self._callback_with_value, callback)
        # for default implementation, connect to editingFinished signal
        self.editingFinished.connect(callback)

    def _callback_with_value(self, callback: callable, *args, **kwargs) -> None:
        """Intermediate callback function, will call user callback with new value

        Parameters
        ----------
        callback : callable
            Callback function to call, with new value as parameter
        """
        # call user callback with new value with correct type
        callback(self.get_value())


class IntControl(ui_widgets.IntSpinBox, WidgetInterface):
    """UI control for integer parameter.

    Parameters
    ----------
    parent : QWidget
        Parent QWidget owning control
    """

    def get_value(self) -> int:
        return self.value()

    def set_value(self, value: int, block_signals=True) -> None:
        if block_signals:
            self.blockSignals(True)
            self.setValue(value)
            self.blockSignals(False)
        else:
            self.setValue(value)


class FloatControl(ui_widgets.FloatSpinBox, WidgetInterface):
    """UI control for float parameter.

    Parameters
    ----------
    parent : QWidget
        Parent QWidget owning control
    """

    def get_value(self) -> float:
        return self.value()

    def set_value(self, value: float, block_signals=True) -> None:
        if block_signals:
            self.blockSignals(True)
            self.setValue(value)
            self.blockSignals(False)
        else:
            self.setValue(value)


class ComplexControl(ui_widgets.ComplexSpinBox, WidgetInterface):
    """UI control for float parameter.

    Parameters
    ----------
    parent : QWidget
        Parent QWidget owning control
    """

    def get_value(self) -> complex:
        return self.value()

    def set_value(self, value: complex, block_signals=True) -> None:
        if block_signals:
            self.blockSignals(True)
            self.setValue(value)
            self.blockSignals(False)
        else:
            self.setValue(value)

    def set_callback(self, callback: callable, with_value=True) -> None:
        # connect to ValueUpdated signal
        if with_value:
            callback = functools.partial(self._callback_with_value, callback)
        self.ValueUpdated.connect(callback)


class BoolControl(QCheckBox, WidgetInterface):
    """UI control for string parameter.

    Parameters
    ----------
    parent : QWidget
        Parent QWidget owning control
    """

    def get_value(self) -> bool:
        return self.isChecked()

    def set_value(self, value: bool, block_signals=True) -> None:
        if block_signals:
            self.blockSignals(True)
            self.setChecked(value)
            self.blockSignals(False)
        else:
            self.setChecked(value)

    def set_callback(self, callback: callable, with_value=True) -> None:
        if with_value:
            callback = functools.partial(self._callback_with_value, callback)
        self.stateChanged[int].connect(callback)


class StringControl(QLineEdit, WidgetInterface):
    """UI control for string parameter.

    Parameters
    ----------
    parent : QWidget
        Parent QWidget owning control
    """

    def get_value(self) -> str:
        return self.text().strip()

    def set_value(self, value: str, block_signals=True) -> None:
        if block_signals:
            self.blockSignals(True)
            self.setText(value)
            self.blockSignals(False)
        else:
            self.setText(value)


class EnumControl(ui_widgets.ComboBox, WidgetInterface):
    """UI control for enum parameter.

    Parameters
    ----------
    parent : QWidget
        Parent QWidget owning control
    """

    def __init__(self, enum_type: Enum, parent: QWidget = None):
        super().__init__(parent)
        # keep track of enum type, add enum values to combobox
        self._enum_type = enum_type
        if self._enum_type is not None:
            self.addItems([e.value for e in self._enum_type])

    def get_value(self) -> Enum:
        s = self.currentText()
        return self._enum_type(s)

    def set_value(self, value: Enum, block_signals=True) -> None:
        n = self.findText(value.value)
        if n >= 0:
            if block_signals:
                self.blockSignals(True)
                self.setCurrentIndex(n)
                self.blockSignals(False)
            else:
                self.setCurrentIndex(n)

    def set_callback(self, callback: callable, with_value=True) -> None:
        if with_value:
            callback = functools.partial(self._callback_with_value, callback)
        self.currentIndexChanged[int].connect(callback)


class PathControl(ui_widgets.PathEdit, WidgetInterface):
    """UI control for path parameter.

    Parameters
    ----------
    parent : QWidget
        Parent QWidget owning control
    """

    def get_value(self) -> str:
        return self.get_path()

    def set_value(self, value: str, block_signals=True) -> None:
        if block_signals:
            self.blockSignals(True)
            self.set_path(value)
            self.blockSignals(False)
        else:
            self.set_path(value)

    def set_callback(self, callback: callable, with_value=True) -> None:
        if with_value:
            callback = functools.partial(self._callback_with_value, callback)
        self.PathChanged.connect(callback)


class DictControl(QGroupBox, WidgetInterface):
    """UI control for a collection parameters, similar to a dict.

    Parameters
    ----------
    collection : dict
        Collection of parameters to display

    parent : QWidget
        Parent QWidget owning control
    """

    def __init__(self, collection: dict, parent: QWidget = None):
        super().__init__(parent)
        self._layout = QGridLayout()
        self._controls = {}
        # create controls for each parameter
        for n, (key, value) in enumerate(collection.items()):
            self._controls[key] = create_control(value, self)
            self._layout.addWidget(QLabel(key), n, 0)
            self._layout.addWidget(self._controls[key], n, 1)
        # format and assign layout to widget
        self._layout.setContentsMargins(6, 6, 6, 4)
        self._layout.setSpacing(2)
        self._layout.setColumnMinimumWidth(0, 100)
        self._layout.setColumnMinimumWidth(1, 120)
        self.setLayout(self._layout)

    def get_value(self) -> dict:
        d = {key: control.get_value() for key, control in self._controls.items()}
        return d

    def set_value(self, value: dict, block_signals=True) -> None:
        for key, data in value.items():
            self._controls[key].set_value(data, block_signals=block_signals)

    def set_callback(self, callback: callable, with_value=True) -> None:
        # update callback for each control
        if with_value:
            callback = functools.partial(self._callback_with_value, callback)
        for control in self._controls.values():
            control.set_callback(callback, with_value=False)


class ListControl(QGroupBox, WidgetInterface):
    """UI control for a list of parameters, representad as a Python list.

    Parameters
    ----------
    value : list
        List containing data of type that the control represents

    parent : QWidget
        Parent QWidget owning control
    """

    def __init__(self, value: list, parent: QWidget = None):
        super().__init__(parent)
        # the list control has two components: a length control and a scroll area
        self._layout = QGridLayout()
        self.scroll_area = _ScrollArea(value[0])
        self.widget_n_elements = IntControl()
        self.widget_n_elements.setRange(0, 1e9)
        self.widget_n_elements.set_callback(self.scroll_area.update_length)
        label = QLabel("Elements")
        self._layout.addWidget(label, 0, 0)
        self._layout.addWidget(self.widget_n_elements, 0, 1)
        self._layout.addWidget(self.scroll_area, 1, 0, 1, 2)
        self._layout.setContentsMargins(6, 6, 6, 4)
        self._layout.setSpacing(2)
        self.setLayout(self._layout)
        # finally, use set value function to update controls
        self.set_value(value)

    def get_value(self) -> list:
        # the internal list in the scroll area should be up-to-date, return copy
        n = self.widget_n_elements.get_value()
        return self.scroll_area.list_data[:n].copy()

    def set_value(self, value: list, block_signals=True) -> None:
        # make copy of list and save in scroll area object, to enable GUI updates
        self.scroll_area.set_list(value.copy())
        self.widget_n_elements.set_value(len(value))

    def set_callback(self, callback: callable, with_value=True) -> None:
        # update callback for each control
        if with_value:
            callback = functools.partial(self._callback_with_value, callback)
        self.scroll_area.list_updated.connect(callback)


class _ScrollArea(QtWidgets.QAbstractScrollArea):
    """Scroll area for handling lists of parameters.

    Parameters
    ----------
    datatype : Any
        Type of element in list
    """

    VERTICAL_SPACING = 2
    # signal emitted when list is updated
    list_updated = QtCore.Signal()

    def __init__(self, datatype):
        super().__init__()
        self._datatype = datatype
        # keep reference to list of values represented by this widget
        self.list_data: list = []
        # number of elements in list, can be fewer than elements in list_data
        self.n_elems: int = 0
        # create layout and widget and set up callbacks
        self._create_layout()
        self.verticalScrollBar().valueChanged.connect(self.on_vscrollbar_value_changed)
        # update scrollbar to redraw scroll area contents
        self.update_scrollbar()

    def _create_layout(self) -> None:
        """Create layout for scroll area."""
        # keep track of labels and subwidgets
        self.labels: list[QLabel] = []
        self.subwidgets: list[QWidget] = []
        # create base widget to determine size
        widget = create_control(self._datatype)
        label = QtWidgets.QLabel("1:")
        # get dimensions of widget, either from size of label or widget
        self.item_height = max(widget.sizeHint().height(), label.sizeHint().height())
        self.item_height += self.VERTICAL_SPACING
        # create layout
        self._layout = QtWidgets.QGridLayout()
        # add spacing
        self._layout.setVerticalSpacing(self.VERTICAL_SPACING)
        self._layout.setSizeConstraint(QtWidgets.QLayout.SetMinAndMaxSize)
        self._layout.setColumnStretch(0, 0)
        self._layout.setColumnStretch(1, 100)
        self._layout.setContentsMargins(0, 0, 0, 0)
        #  create main widget in viewport
        self.main_widget = QtWidgets.QGroupBox(self.viewport())
        self.main_widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.main_widget.setLayout(self._layout)
        # set main widget sizehint to be same as viewport size, to keep fixed width
        self.main_widget.sizeHint = lambda: QtCore.QSize(
            self.viewport().width(), self.viewport().height()
        )

    def set_list(self, value: list, n_elems: int = None) -> None:
        """Set list represented by scroll area

        Parameters
        ----------
        value : list
            List with values
        n_elems : int, optional
            Length of list.  If None, determine from `value` parameter.
        """
        self.list_data = value
        # update number of elements
        if n_elems is None:
            n_elems = len(value)
        self.n_elems = n_elems
        self.update_scrollbar()

    def on_vscrollbar_value_changed(self, value) -> None:
        """Callback for scrollbar updates

        Parameters
        ----------
        value : int
            New postition of scrollbar.
        """
        # if in edit mode, update value and clear focus before starting scrolling
        widget = QtWidgets.QApplication.focusWidget()
        if widget in self.subwidgets:
            n = self.subwidgets.index(widget)
            self.callback_value(n, widget.get_value())
            widget.clearFocus()
        # move main widget to new position modulues item height, for small shifts
        self.main_widget.move(0, -(value % self.item_height))
        # redraw label and widgets to accomodate for shifts larger than one item
        self.update_widgets()

    def update_scrollbar(self) -> None:
        """Update scrollbar so that the range matches number of elements in list."""
        self.verticalScrollBar().setRange(
            0, self.n_elems * self.item_height - self.viewport().height()
        )
        self.verticalScrollBar().setPageStep(self.viewport().height())
        self.update_widgets()

    def update_length(self, n_elements) -> None:
        """Update length of list, adding empty elements if needed."""
        old_n_elements = len(self.list_data)
        if n_elements > old_n_elements:
            # add elements to list
            self.list_data.extend(
                [type(self.list_data[0])()] * (n_elements - old_n_elements)
            )
        self.set_list(self.list_data, n_elements)

    def update_widgets(self) -> None:
        """Update labels and widgets in scroll area to match scrollbar position"""
        # calculate number of widgets that can partially fit in viewport
        n_visible = 2 + self.viewport().height() // self.item_height
        # only create as many as needed - list length or number visible in scroll area
        n_widgets = min(n_visible, self.n_elems)
        # create more widgets if needed
        for n in range(len(self.subwidgets), n_widgets):
            widget = create_control(self._datatype)
            widget.set_callback(functools.partial(self.callback_value, n))
            self.subwidgets.append(widget)
            self.labels.append(QtWidgets.QLabel(""))
            self._layout.addWidget(self.labels[-1], n, 0)
            self._layout.addWidget(self.subwidgets[-1], n, 1)
        # determine which list elements are visible
        first = self.verticalScrollBar().value() // self.item_height
        # update labels and widget values from internal list
        for n, (label, widget) in enumerate(zip(self.labels, self.subwidgets)):
            if (n + first) < self.n_elems:
                # item active, update text and show widgets
                label.setText(f"{n + first}:")
                widget.set_value(self.list_data[n + first])
                label.show()
                widget.show()
            else:
                # not enough active elements, hide widgets
                label.hide()
                widget.hide()

    def resizeEvent(self, event):
        """Resize event for scroll area, update size of main widget to match viewport"""
        # update width of main widget to match viewport
        self.main_widget.setFixedWidth(self.viewport().width())
        self.update_scrollbar()
        super().resizeEvent(event)

    def callback_value(self, index, value):
        """Callback for when a subwidget value is changed"""
        # update correct element in list
        first = self.verticalScrollBar().value() // self.item_height
        self.list_data[first + index] = value
        # emit signal to main list control to signal new value
        self.list_updated.emit()
