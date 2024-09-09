"""QT widgets to be used in Polaris graphical user interface."""

import functools
import os
import re
import sys
from enum import Enum
from typing import Callable

import numpy as np
from qtpy import QtCore
from qtpy.compat import getexistingdirectory, getopenfilename, getsavefilename
from qtpy.QtCore import QSize, Qt, QTime, Signal
from qtpy.QtGui import (
    QColor,
    QFont,
    QIcon,
    QKeySequence,
    QPixmap,
    QValidator,
    QWheelEvent,
)
from qtpy.QtWidgets import (
    QAbstractSpinBox,
    QAction,
    QActionGroup,
    QColorDialog,
    QComboBox,
    QDateTimeEdit,
    QDoubleSpinBox,
    QFontComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPlainTextEdit,
    QSizePolicy,
    QTableView,
    QTimeEdit,
    QToolButton,
    QWidget,
)

# resource file contains icons, needs to be imported
from . import ui_resources  # noqa: F401 # pylint:disable=unused-import
from . import str_helper, ui_tools
from .ui_tools import MAC, get_theme_icon


class ComboBox(QComboBox):
    """Combo box implementation without mouse wheel scrolling"""

    def __init__(self, parent=None):
        super().__init__(parent)
        # show longer list of items by default
        self.setMaxVisibleItems(40)
        self.setFocusPolicy(Qt.StrongFocus)

    def wheelEvent(self, event):
        QWheelEvent.ignore(event)


class DisplayFormat(Enum):
    """Enum for number format description"""

    FLOAT = "Float"
    ENG = "Engineering"
    EXP = "Exponential"
    SI = "SI prefix"


class FloatSpinBox(QDoubleSpinBox):
    """A spinbox with SI exponential and engineering formatting capabilities.

    THe number of digits and other formatting options can be set at
    initialization or by the user via a user context menu.

    Parameters
    ----------
    parent : QWidget, optional
        Parent widget, by default None
    display_format : DisplayFormat, optional
        Initial display format, by default DisplayFormat.ENG
    digits : int, optional
        Number of significant digits, by default 4
    """

    def __init__(
        self,
        parent: QWidget = None,
        display_format: DisplayFormat = DisplayFormat.ENG,
        digits=4,
    ):
        super().__init__(parent)
        # init instance variables
        self.display_format = display_format
        self.digits = digits
        # set widget properties
        self.setAlignment(Qt.AlignRight)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setDecimals(99)
        self.setRange(-float("inf"), float("inf"))
        # size of change with keyboard up/down
        self.step_up_down = None
        self.page_up_down = None
        # reset up/down step size upon entering new value
        self.lineEdit().textEdited["QString"].connect(self.reset_up_down_step)
        # create context menu
        self.create_context_menu()

    def create_context_menu(self) -> None:
        """Create context menu with formatting options"""
        self.setContextMenuPolicy(Qt.ActionsContextMenu)
        # standard cut/copy/paste context actions
        cut_action = QAction("Cut", self)
        cut_action.triggered.connect(self.cut_text)
        copy_action = QAction("Copy", self)
        copy_action.triggered.connect(self.copy_text)
        paste_action = QAction("Paste", self)
        paste_action.triggered.connect(self.paste_text)
        del_action = QAction("Delete", self)
        del_action.triggered.connect(self.clear)
        select_all_action = QAction("Select All", self)
        select_all_action.triggered.connect(self.selectAll)
        # submenu with format setting
        format_action = QAction("Format", self)
        # subgroup with format type settings
        format_group = QActionGroup(self)
        format_actions = []
        # create action for each formatting type
        for x in DisplayFormat:
            action = ui_tools.create_action(
                format_group,
                x.value,
                functools.partial(self.set_format, x),
                checkable=True,
            )
            action.setChecked(x == self.display_format)
            format_actions.append(action)
        # subgroup with digits of precision
        digits_label = QAction("Significant digits", self)
        digits_label.setEnabled(False)
        precision_group = QActionGroup(self)
        digits_actions = [
            ui_tools.create_action(
                precision_group,
                f"{n1 + 1}",
                functools.partial(self.set_digits, n1 + 1),
                checkable=True,
            )
            for n1 in range(9)
        ]
        # mark the correct digits as active
        digits_actions[self.digits - 1].setChecked(True)
        # add all actions to the format submenu
        format_menu = QMenu(self)
        ui_tools.add_actions(format_menu, format_actions)
        ui_tools.add_actions(format_menu, (None, digits_label))
        ui_tools.add_actions(format_menu, digits_actions)
        format_action.setMenu(format_menu)
        # only add formatting actions if digits are allowed (otherwise int)
        if self.digits > 0:
            ui_tools.add_actions(self, (format_action, None))
        # add actions to control to create the context menu
        ui_tools.add_actions(self, (cut_action, copy_action, paste_action))
        ui_tools.add_actions(self, (del_action, None, select_all_action))

    def set_digits(self, digits: int, *extra_args, select_all=True):
        """Set digits of precision

        Parameters
        ----------
        digits : int
            Digits of precision
        select_all : bool, optional
            Select all text after finishing operation, by default True
        """
        self.digits = digits
        self.setValue(self.value())
        # mark all when changing unit
        if select_all:
            self.selectAll()

    def set_format(self, display_format: DisplayFormat, *extra_args):
        """Set display format

        Parameters
        ----------
        display_format : DisplayFormat
            New display format
        """
        self.display_format = display_format
        self.setValue(self.value())
        self.selectAll()

    def cut_text(self):
        """Cut text in dialog (as string, not float)"""
        text_widget = self.lineEdit()
        text_widget.cut()

    def copy_text(self):
        """Copy text in dialog (as string, not float)"""
        text_widget = self.lineEdit()
        text_widget.copy()

    def paste_text(self):
        """Paste text in dialog (as string, not float)"""
        text_widget = self.lineEdit()
        text_widget.paste()

    def reset_up_down_step(self, *extra_args):
        """Reset up/down step size, called when entering new value"""
        # reset the up/down step size
        self.step_up_down = None
        self.page_up_down = None

    # the following functions are overloaded Qt functions
    def stepBy(self, steps):
        # check if step size needs to be recalculated
        if self.step_up_down is None:
            # find first/last significant digit from text in box
            value = self.value().real
            if self.display_format == DisplayFormat.FLOAT:
                str_value = f"{value:#.{self.digits}f}"
                str_value = str_value.rstrip("0").rstrip(".")
            elif self.display_format == DisplayFormat.EXP:
                str_value = f"{value:.{self.digits - 1}e}".upper()
            else:
                str_value = str_helper.get_engineering_str(value, decimals=self.digits)

            # split at exponential, replace with zeros, strip -/+ signs
            n = str_value.find("E")
            if n < 0:
                n = len(str_value)
            num_zero = re.sub("[0-9]", "0", str_value[: n - 1])
            num_zero = re.sub(r"-|\+", "", num_zero)
            # add 1 to create least significant string, add exponential part
            str_least = num_zero + "1" + str_value[n:]
            try:
                self.step_up_down = abs(float(str_least))
            except Exception:
                # return directly if conversion fails
                return

            # same for most significant digit
            num_zero = re.sub("[0-9]", "0", str_value[:n])
            num_zero = re.sub(r"-|\+", "", num_zero)
            str_most = "1" + num_zero + str_value[n:]
            try:
                self.page_up_down = abs(float(str_most) / 10)
            except Exception:
                # return directly if conversion fails
                return
            # round value to forget invisible digits
            value = self.valueFromText(self.lineEdit().text())
        else:
            value = self.value()
        # check if up/down or page up/page down
        if abs(steps) > 1:
            step_size = self.page_up_down
        else:
            step_size = self.step_up_down
        # apply changes
        self.setValue(value + step_size * np.sign(steps))
        self.selectAll()
        self.editingFinished.emit()

    def wheelEvent(self, event):
        QWheelEvent.ignore(event)

    def validate(self, text, pos):
        return (QValidator.Acceptable, text, pos)

    def valueFromText(self, text):
        value_str = str(text).upper().strip()
        if value_str == "INF":
            return float("inf")
        if value_str == "-INF":
            return float("-inf")
        if value_str == "NAN":
            # encode NaN as min float value
            return sys.float_info.min
        value = str_helper.get_value_from_si_string(str(text))
        return 0.0 if (value is None) else value

    def textFromValue(self, value):
        # NaN is encoded as min float value
        if value == sys.float_info.min:
            return "NaN"
        if value == float("inf"):
            return "Inf"
        if value == float("-inf"):
            return "-Inf"
        if self.display_format == DisplayFormat.FLOAT:
            # convert from float, keep decimal point to not trim wrong zeros
            x = f"{value:#.{self.digits}f}"
            # remove trailing zeros and trailing decimal point
            x = x.rstrip("0").rstrip(".")
            return x.upper()
        if self.display_format == DisplayFormat.ENG:
            return str_helper.get_engineering_str(value, decimals=self.digits)
        if self.display_format == DisplayFormat.EXP:
            return f"{value:.{self.digits - 1}e}".upper()
        if self.display_format == DisplayFormat.SI:
            return str_helper.get_si_string(value, decimals=self.digits)

    def value(self):
        # take care of NaN, encoded as min float value
        value = super().value()
        if value == sys.float_info.min:
            return float("nan")
        return value


class IntSpinBox(FloatSpinBox):
    """An int spinbox with SI, exp and engineering formatting capabilities.

    The number of digits and other formatting options can be set at
    initialization or by the user via a user context menu.

    Parameters
    ----------
    parent : QWidget, optional
        Parent widget, by default None
    """

    def __init__(self, parent: QWidget = None):
        # initialize base class as float with zero digits precision
        super().__init__(parent, DisplayFormat.FLOAT, digits=0)

    def value(self):
        # get value as float from base float spinbox class
        value = super().value()
        # try to convert to int, return 0 if failure
        try:
            return int(value)
        except Exception:
            return 0

    def valueFromText(self, text):
        # get value as float from base class, convert to int, return 0 if failure
        value = super().valueFromText(text)
        try:
            return int(value)
        except Exception:
            return 0


class ComplexView(Enum):
    """Enum for complex number representation"""

    REALIMAG = "Real - Imaginary"
    MAGPHASE = "Magnitude - Phase"


class ComplexSpinBox(QAbstractSpinBox):
    """A complex spinbox with exponential and engineering formatting."""

    # define update signal
    ValueUpdated = Signal()

    def __init__(
        self,
        parent: QWidget = None,
        display_format: DisplayFormat = DisplayFormat.ENG,
        digits=4,
        complex_view=ComplexView.REALIMAG,
    ):
        super().__init__(parent)
        # init instance variables
        self.display_format = display_format
        self.digits = digits
        self.complex_view = complex_view
        self._value = 0.0 + 0.0j
        # set widget properties
        self.setAlignment(Qt.AlignRight)
        self.setFocusPolicy(Qt.StrongFocus)
        self.create_context_menu()
        # connect editing finished signal from lineEdit
        self.lineEdit().editingFinished.connect(self.editFinished)
        self.lineEdit().returnPressed.connect(self.editFinished)
        # initialize dialog value
        self.editFinished()

    def create_context_menu(self) -> None:
        """Create context menu with formatting options"""
        self.setContextMenuPolicy(Qt.ActionsContextMenu)
        # standard cut/copy/paste context actions
        cut_action = QAction("Cut", self)
        cut_action.triggered.connect(self.cut_text)
        copy_action = QAction("Copy", self)
        copy_action.triggered.connect(self.copy_text)
        paste_action = QAction("Paste", self)
        paste_action.triggered.connect(self.paste_text)
        del_action = QAction("Delete", self)
        del_action.triggered.connect(self.clear)
        select_all_action = QAction("Select All", self)
        select_all_action.triggered.connect(self.selectAll)
        # submenu with format setting
        format_action = QAction("Format", self)
        # subgroup with format type settings
        format_group = QActionGroup(self)
        format_actions = []
        # create action for each formatting type
        for x in DisplayFormat:
            if x == DisplayFormat.SI:
                # SI format not supported for complex numbers
                continue
            action = ui_tools.create_action(
                format_group,
                x.value,
                functools.partial(self.set_format, x),
                checkable=True,
            )
            action.setChecked(x == self.display_format)
            format_actions.append(action)
        # subgroup with digits of precision
        digits_label = QAction("Significant digits", self)
        digits_label.setEnabled(False)
        precision_group = QActionGroup(self)
        digits_actions = [
            ui_tools.create_action(
                precision_group,
                f"{n1 + 1}",
                functools.partial(self.set_digits, n1 + 1),
                checkable=True,
            )
            for n1 in range(9)
        ]
        # mark the correct digits as active
        digits_actions[self.digits - 1].setChecked(True)
        # add all actions to the format submenu
        format_menu = QMenu(self)
        ui_tools.add_actions(format_menu, format_actions)
        ui_tools.add_actions(format_menu, (None, digits_label))
        ui_tools.add_actions(format_menu, digits_actions)
        format_action.setMenu(format_menu)

        # create complex format settings
        complex_group = QActionGroup(self)
        complex_label = QAction("Complex view", self)
        complex_label.setEnabled(False)
        complex_actions = [complex_label]
        for x in ComplexView:
            action = ui_tools.create_action(
                complex_group,
                x.value,
                functools.partial(self.set_complex_view, x),
                checkable=True,
            )
            action.setChecked(x == self.complex_view)
            complex_actions.append(action)

        # add actions to control to create the context menu
        ui_tools.add_actions(self, complex_actions)
        ui_tools.add_actions(self, (None, format_action, None, cut_action, copy_action))
        ui_tools.add_actions(self, (paste_action, del_action, None, select_all_action))

    def set_digits(self, digits: int, *extra_args, select_all=True):
        """Set digits of precision

        Parameters
        ----------
        digits : int
            Digits of precision
        select_all : bool, optional
            Select all text after finishing operation, by default True
        """
        self.digits = digits
        self.setValue(self.value())
        # mark all when changing unit
        if select_all:
            self.selectAll()

    def set_format(self, display_format: DisplayFormat, *extra_args):
        """Set display format

        Parameters
        ----------
        display_format : DisplayFormat
            New display format
        """
        self.display_format = display_format
        self.setValue(self.value())
        self.selectAll()

    def cut_text(self):
        """Cut text in dialog (as string, not float)"""
        text_widget = self.lineEdit()
        text_widget.cut()

    def copy_text(self):
        """Copy text in dialog (as string, not float)"""
        text_widget = self.lineEdit()
        text_widget.copy()

    def paste_text(self):
        """Paste text in dialog (as string, not float)"""
        text_widget = self.lineEdit()
        text_widget.paste()

    def set_complex_view(self, complex_view: ComplexView, *extra_args) -> None:
        """Set complex view

        Parameters
        ----------
        complex_view : ComplexView
            New complex view
        """
        self.complex_view = complex_view
        self.setValue(self.value())
        self.selectAll()

    def editFinished(self):
        # overload of QT function
        value = self.valueFromText(self.lineEdit().text())
        self.setValue(value)
        self.ValueUpdated.emit()

    def setValue(self, value):
        # overload of QT function
        self._value = value
        self.lineEdit().setText(self.textFromValue(value))

    def value(self):
        # overload of QT function
        return self._value

    def valueFromText(self, text):
        # overload of QT function
        # make sure formatting is clean
        text = str(text).strip()
        # check if angle argument is present in text (<)
        n = text.find("<")
        try:
            if n > 0:
                # angle is given, split the two parts
                mag = str_helper.get_value_from_si_string(text[:n])
                if mag is None:
                    raise ValueError()
                angle = float(text[n + 1 :].strip())
                return complex(mag * np.exp(1j * angle * np.pi / 180))
            # complex, replace INF with QNF to avoid mistaking i in inf
            text_org = text
            text = text.lower()
            text = text.replace("inf", "qnf")
            # next, make I to J and convert combined nJ into n*1J (upper case j)
            text = text.replace("i", "j")
            text = text.replace("0j", "0*1J")
            text = text.replace("1j", "1*1J")
            text = text.replace("2j", "2*1J")
            text = text.replace("3j", "3*1J")
            text = text.replace("4j", "4*1J")
            text = text.replace("5j", "5*1J")
            text = text.replace("6j", "6*1J")
            text = text.replace("7j", "7*1J")
            text = text.replace("8j", "8*1J")
            text = text.replace("9j", "9*1J")
            text = text.replace(".j", ".*1J")
            # replace remaining single Js with 1j
            text = text.replace("j", "1J")
            # finally, re-replace QNF with INF
            text = text.replace("qnf", "inf")
            if text.find("1J") < 0:
                # no complex part, try to convert to float using SI
                value = str_helper.get_value_from_si_string(text_org)
                if value is None:
                    raise ValueError()
                return complex(value)
            # fix infinity and NANs
            text = text.replace("inf", "np.inf")
            text = text.replace("nan", "np.nan")
            # use eval to convert to complex number
            value = eval(f"complex({text})")
            return value
        except (ValueError, SyntaxError, NameError):
            return complex(0)

    def textFromValue(self, value):
        # check output format
        if self.complex_view == ComplexView.REALIMAG:
            # get strings for real and complex parts
            value_strings = []
            for x in [value.real, value.imag]:
                if self.display_format == DisplayFormat.FLOAT:
                    s = f"{x:#.{self.digits}f}"
                    s = s.rstrip("0").rstrip(".").upper()
                elif self.display_format == DisplayFormat.EXP:
                    s = f"{x:.{self.digits - 1}e}".upper()
                elif self.display_format in (DisplayFormat.ENG, DisplayFormat.SI):
                    s = str_helper.convert_to_si(x, decimals=self.digits)["exp_str"]
                value_strings.append(s)
            # fix pos/neg signs for imaginary part
            if value.imag < 0:
                value_strings[-1] = " - " + value_strings[-1][1:]
            else:
                value_strings[-1] = " + " + value_strings[-1]
            # construct complex string
            return value_strings[0] + value_strings[1] + "j"
        if self.complex_view == ComplexView.MAGPHASE:
            # magnitud-phase, get numbers
            mag = abs(value)
            angle = np.angle(value) * 180.0 / np.pi
            # create string for magnitude
            if self.display_format == DisplayFormat.FLOAT:
                s = f"{mag:#.{self.digits}f}"
                s = s.rstrip("0").rstrip(".").upper()
            elif self.display_format == DisplayFormat.EXP:
                s = f"{mag:.{self.digits - 1}e}".upper()
            elif self.display_format in (DisplayFormat.ENG, DisplayFormat.SI):
                s = str_helper.convert_to_si(mag, decimals=self.digits)["exp_str"]
            # add float for angle
            return f"{s} < {angle:.1f}"


class ColorButton(QToolButton):
    """A color button, shows a color input dialog when clicked

    Parameters
    ----------
    color : str | QColor, optional
        New color, by default 'red'
    parent : QWidget, optional
        Parent widget, by default None
    """

    # define update signal
    ColorHasChanged = Signal()

    @staticmethod
    def get_color_from_string(color: str | QColor) -> QColor:
        """Convert python/matlab color string to QT color

        Parameters
        ----------
        color : str | QColor
            New color

        Returns
        -------
        QColor
            Color in QT format
        """

        # expand matlab/python-type color codes
        colors = dict(
            r="red",
            b="blue",
            g="green",
            c="cyan",
            m="magenta",
            y="yellow",
            k="black",
            w="white",
        )
        # update color from conversion dict, if avaliable
        color = colors.get(color, color)
        return QColor(color)

    def __init__(self, color: str | QColor = "red", parent: QWidget = None):
        super().__init__(parent)
        # create QColor from string
        color = ColorButton.get_color_from_string(color)
        if not color.isValid():
            color = QColor("black")
        self.set_color(color)
        # set callback to open color dialog
        self.clicked.connect(self.show_color_dialog)
        self.setMaximumWidth(50)
        self.setMaximumHeight(16)

    def show_color_dialog(self) -> None:
        """Show dialog for picking new color"""
        new_color = QColorDialog.getColor(self.color)
        if new_color.isValid():
            self.set_color(new_color)
            self.ColorHasChanged.emit()

    def set_color(self, color: str | QColor) -> None:
        """Set new color of button

        Parameters
        ----------
        color : str | QColor
            New color
        """
        if isinstance(color, str):
            color = ColorButton.get_color_from_string(color)
        self.color = color
        self.setStyleSheet(f"background-color:{self.color.name()}")

    def get_color_as_string(self) -> str:
        """Return color as hexadecimal string

        Returns
        -------
        str
            Color as hexadecimal string
        """
        return str(self.color.name())


class ConfigButton(QToolButton):
    """A config button for opening a settings dialog"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setIconSize(QSize(18, 12))
        self.setIcon(QIcon(get_theme_icon(":/settings")))


class FolderButton(QToolButton):
    """A folder button for opening a file dialog window"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setIconSize(QSize(18, 12))
        self.setIcon(QIcon(get_theme_icon(":/folder")))


class PlainTextEdit(QPlainTextEdit):
    """Plain text edit that emits a signal when losing focus"""

    # define signal
    editingFinished = Signal()

    def __init__(self, string="", parent=None):
        super().__init__(string, parent=parent)
        self.setTabChangesFocus(True)

    def focusOutEvent(self, event):
        # emit signal when loosing focus
        self.editingFinished.emit()
        QPlainTextEdit.focusOutEvent(self, event)


class PathEdit(QWidget):
    """A folder button together with a path name display widget

    Parameters
    ----------
    path_name : str, optional
        Start path, by default ''
    user_prompt : str, optional
        User prompt, by default 'Select a file'
    path_filter : str, optional
        Path filter for selecting files, by default ''
    select_folder : bool, optional
        If True, dialog will select a folder, not a file. By default False
    for_saving : bool, optional
        If True, the dialog will be used for saving a file, by default False
    callback : Callable, optional
        Callback on file change, by default None
    single_line : bool, optional
        If True, dialog will fit on single line, by default True
    normalize_path : bool, optional
        If True, path will be normalized to match OS, by default True
    parent : _type_, optional
        Parent widget, by default None
    """

    # define signals
    PathChanged = Signal(str)

    def __init__(
        self,
        path_name: str = "",
        user_prompt="Select a file",
        path_filter: str = "",
        select_folder=False,
        for_saving=False,
        callback: Callable = None,
        single_line=True,
        normalize_path=True,
        parent=None,
    ):
        super().__init__(parent)
        # save settings
        self.path_name = path_name
        self.user_prompt = user_prompt
        self.path_filter = path_filter
        self.select_folder = select_folder
        self.for_saving = for_saving
        self.single_line = single_line
        self.normalize_path = normalize_path

        self.button = FolderButton()
        # connect provided button callback, else use default
        if callback is None:
            callback = self.button_callback
        self.button.clicked.connect(callback)
        if single_line:
            self.line_edit = QLineEdit(path_name)
        else:
            self.line_edit = PlainTextEdit(path_name)
        self.line_edit.editingFinished.connect(self.line_edit_callback)
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.line_edit)
        layout.addWidget(self.button)
        if single_line:
            s = self.sizePolicy()
            s.setVerticalPolicy(QSizePolicy.Fixed)
            self.setSizePolicy(s)
        else:
            layout.setAlignment(self.button, Qt.AlignTop)
        self.setLayout(layout)

    def set_path(self, path_name: str) -> None:
        """Set the current path value by updating the line edit control

        Parameters
        ----------
        path_name : str
            New path
        """
        # normalize path, but avoid converting empty to "."
        if self.normalize_path:
            path_name = os.path.normpath(path_name)
            path_name = "" if path_name == "." else path_name
        if self.single_line:
            self.line_edit.setText(path_name)
        else:
            self.line_edit.setPlainText(path_name)
        self.path_name = path_name

    def get_path(self) -> str:
        """Get the current path value

        Returns
        -------
        str
            Current path name
        """
        if self.single_line:
            path_name = str(self.line_edit.text()).strip()
        else:
            path_name = str(self.line_edit.toPlainText()).strip()
        # normalize path, but avoid converting empty to "."
        if self.normalize_path:
            path_name = os.path.normpath(path_name)
            path_name = "" if path_name == "." else path_name
        return path_name

    def button_callback(self) -> None:
        """Callback for folder button"""
        path_name = self.get_path()
        if self.select_folder:
            fname = getexistingdirectory(self, self.user_prompt, path_name)
        elif self.for_saving:
            fname = getsavefilename(self, self.user_prompt, path_name, self.path_filter)
        else:
            fname = getopenfilename(self, self.user_prompt, path_name, self.path_filter)
        fname = fname[0] if isinstance(fname, tuple) else fname
        # check if user cancelled the dialog
        if len(fname) == 0:
            return False
        # normalize path, but avoid converting empty to "."
        new_path = str(fname)
        if self.normalize_path:
            new_path = os.path.normpath(new_path)
            new_path = "" if new_path == "." else new_path
        self.set_path(new_path)
        self.PathChanged.emit(new_path)

    def line_edit_callback(self) -> None:
        """Callback after the line edit text has changed"""
        new_path = self.get_path()
        # if path changed, emit signal that folder has changed
        if new_path != self.path_name:
            self.path_name = new_path
            self.PathChanged.emit(new_path)


class FontEdit(QWidget):
    """A font viewer/editor with button for showing a font dialog

    Parameters
    ----------
    font : str | QFont
        Font in use
    single_line : bool, optional
        If True, control fits on single line, by default False
    parent : QWidget, optional
        Parent widget, by default None
    """

    # define signals
    FontUpdated = Signal()

    def __init__(
        self, font: str | QFont = "", single_line=False, parent: QWidget = None
    ):
        super().__init__(parent)
        # create layout
        self.combo_font = QFontComboBox()
        self.combo_font.setFontFilters(self.combo_font.FontFilter.ScalableFonts)
        self.combo_font.setMinimumWidth(50)
        # bold/italic
        self.tool_bold = QToolButton()
        self.tool_bold.setIconSize(QSize(18, 12))
        self.tool_bold.setIcon(QIcon(get_theme_icon(":/text-bold")))
        self.tool_bold.setCheckable(True)
        self.tool_italic = QToolButton()
        self.tool_italic.setIconSize(QSize(18, 12))
        self.tool_italic.setIcon(QIcon(get_theme_icon(":/text-italic")))
        self.tool_italic.setCheckable(True)
        # size
        self.spin_size = FloatSpinBox(digits=9, display_format=DisplayFormat.FLOAT)
        # callbacks
        self.combo_font.currentFontChanged["QFont"].connect(self.tool_button_callback)
        self.tool_bold.clicked.connect(self.tool_button_callback)
        self.tool_bold.clicked.connect(self.tool_button_callback)
        self.tool_italic.clicked.connect(self.tool_button_callback)
        self.spin_size.editingFinished.connect(self.tool_button_callback)
        if single_line:
            layout = QHBoxLayout()
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(1)
            layout.addWidget(self.combo_font)
            layout.addWidget(self.tool_bold)
            layout.addWidget(self.tool_italic)
            layout.addWidget(self.spin_size)
        else:
            layout = QGridLayout()
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setHorizontalSpacing(1)
            layout.setVerticalSpacing(1)
            layout.addWidget(self.combo_font, 0, 0, 1, 5)
            layout.addWidget(self.tool_bold, 1, 0)
            layout.addWidget(self.tool_italic, 1, 1)
            layout.addWidget(QLabel("Size: "), 1, 3)
            layout.addWidget(self.spin_size, 1, 4)
            layout.setColumnStretch(2, 100)
        self.setLayout(layout)
        # set font
        self.set_font(font)

    def set_font(self, font: str | QFont) -> None:
        """Set the current font and update line

        Parameters
        ----------
        font : str | QFont
            New font
        """
        # create font
        if isinstance(font, str):
            self.font = QFont()
            self.font.fromString(font)
        else:
            self.font = QFont(font)
        # set controls
        self.combo_font.blockSignals(True)
        self.combo_font.setCurrentFont(self.font)
        self.combo_font.blockSignals(False)
        self.tool_bold.blockSignals(True)
        self.tool_bold.setChecked(self.font.bold())
        self.tool_bold.blockSignals(False)
        self.tool_italic.blockSignals(True)
        self.tool_italic.setChecked(self.font.italic())
        self.tool_italic.blockSignals(False)
        self.spin_size.blockSignals(True)
        self.spin_size.setValue(self.font.pointSize())
        self.spin_size.blockSignals(False)

    def get_font(self) -> QFont:
        """Get the current font value

        Returns
        -------
        QFont
            Font in use
        """
        return QFont(self.font)

    def get_font_as_string(self) -> str:
        """Get the current font as string

        Returns
        -------
        str
            Font in use as string
        """
        return str(self.font.toString())

    def tool_button_callback(self) -> None:
        """Callback for toolbuttons"""
        # update font settings
        self.font = self.combo_font.currentFont()
        self.font.setPointSize(int(self.spin_size.value()))
        self.font.setBold(self.tool_bold.isChecked())
        self.font.setItalic(self.tool_italic.isChecked())
        self.FontUpdated.emit()


class LED(QLabel):
    """A LED depicting the state of a boolean

    Parameters
    ----------
    size : int, optional
        LED size in pixels, by default 18
    parent : QWidget, optional
        Parent widget, by default None
    """

    @staticmethod
    def get_icon(state=False) -> QIcon:
        """Create a LED icon in on or off state

        Parameters
        ----------
        state : bool, optional
            LED state, by default False

        Returns
        -------
        QIcon
            New LED icon
        """
        icon = QIcon()
        if state:
            icon.addPixmap(
                QPixmap(get_theme_icon(":/led-on")).scaled(64, 64),
                state=QIcon.Off,
            )
        else:
            icon.addPixmap(
                QPixmap(get_theme_icon(":/led-off")).scaled(64, 64),
                state=QIcon.Off,
            )
        return icon

    def __init__(self, size: int = 18, parent=None):
        super().__init__(parent)
        self.size = QSize(size, size)
        self.icon_led = QIcon()
        self.icon_led.addPixmap(
            QPixmap(get_theme_icon(":/led-on")).scaled(self.size), state=QIcon.On
        )
        self.icon_led.addPixmap(
            QPixmap(get_theme_icon(":/led-off")).scaled(self.size), state=QIcon.Off
        )
        # just a LED, no text
        self.setPixmap(self.icon_led.pixmap(self.size, state=QIcon.Off))

    def set_state(self, state=False) -> None:
        """Set the state of the LED light

        Parameters
        ----------
        state : bool, optional
            LED state, by default False
        """
        if state:
            iconState = QIcon.On
        else:
            iconState = QIcon.Off
        self.setPixmap(self.icon_led.pixmap(self.size, state=iconState))


class DateTimeEdit(QDateTimeEdit):
    """A date/time edit with calendar pop-up enabled by default"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # set default display format and calendar pop-up
        self.setDisplayFormat("yyyy-MM-dd, hh:mm:ss")
        self.setCalendarPopup(True)


class TimeEdit(QTimeEdit):
    """Time control with context menu for setting time format"""

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        # set alignment
        self.setAlignment(Qt.AlignRight)
        # create context menu
        self.setContextMenuPolicy(Qt.ActionsContextMenu)
        # add cut/copy/paste context actions
        cut = ui_tools.create_action(
            self, "Cut", self.cut_text, shortcut=QKeySequence.Cut
        )
        copy = ui_tools.create_action(
            self, "Copy", self.copy_text, shortcut=QKeySequence.Copy
        )
        paste = ui_tools.create_action(
            self, "Paste", self.paste_text, shortcut=QKeySequence.Paste
        )
        sel_all = ui_tools.create_action(
            self, "Select All", self.selectAll, shortcut=QKeySequence.SelectAll
        )
        delete = ui_tools.create_action(self, "Delete", self.clear)
        # milliseconds
        self.action_ms = ui_tools.create_action(
            self, "Show milliseconds", self.show_ms, checkable=True
        )
        ui_tools.add_actions(
            self, (self.action_ms, None, cut, copy, paste, delete, None, sel_all)
        )
        self.setTimeRange(QTime(0, 0), QTime(99, 0))
        # show/hide milliseconds
        self.show_ms()

    def show_ms(self, show: bool | None = None) -> None:
        """Show/hide millisecond information

        Parameters
        ----------
        show : bool | None, optional
            If True, show milliseconds, by default None (=use current state)
        """
        # get current state if not given
        if show is None:
            show = self.action_ms.isChecked()
        else:
            self.action_ms.setChecked(show)
        if show:
            self.setDisplayFormat("hh:mm:ss.zzz")
        else:
            self.setDisplayFormat("hh:mm:ss")

    def cut_text(self) -> None:
        """Cut text"""
        line_edit = self.lineEdit()
        line_edit.cut()

    def copy_text(self) -> None:
        """Copy text"""
        line_edit = self.lineEdit()
        line_edit.copy()

    def paste_text(self) -> None:
        """Paste text"""
        line_edit = self.lineEdit()
        line_edit.paste()


class ListSelectionView(QTableView):
    """View widget for a row-based tables with multiple columns

    Parameters
    ----------
    multi_selection : bool, optional
        Allow multiple rows to be selected, by default False
    parent : QObject, optional
        Parent object, by default None
    """

    def __init__(self, multi_selection=False, parent=None):
        super(ListSelectionView, self).__init__(parent)
        # remove vertical headers and grid
        self.verticalHeader().setMinimumSectionSize(18)
        self.verticalHeader().setDefaultSectionSize(18)
        self.verticalHeader().setVisible(False)
        self.setShowGrid(False)
        # set selection mode
        if multi_selection:
            self.setSelectionMode(self.SelectionMode.ExtendedSelection)
        else:
            self.setSelectionMode(self.SelectionMode.SingleSelection)
        self.setSelectionBehavior(self.SelectionBehavior.SelectRows)
        # settings for header
        self.horizontalHeader().setVisible(True)
        self.horizontalHeader().setHighlightSections(False)
        self.horizontalHeader().setStretchLastSection(True)
        # fix font size issue on mac
        if MAC and True:
            font = self.font()
            font.setPointSize(12)
            self.setFont(font)

    def get_selected_rows(self) -> list[int]:
        """Return the selected rows as a list of integers

        Returns
        -------
        list[int]
            Selected rows
        """
        rows = [x.row() for x in self.selectionModel().selectedRows()]
        rows.sort()
        return rows

    def select_rows(self, rows: list[int]) -> None:
        """Set selected rows

        Note that only the first row will be selected for single-selection tables

        Parameters
        ----------
        rows : list[int]
            Rows to select.
        """
        # block events while updating
        model = self.selectionModel()
        model.blockSignals(True)
        # create selection
        for n, row in enumerate(rows):
            self.selectRow(row)
            if n == 0:
                selection = model.selection()
            else:
                selection.merge(model.selection(), QtCore.QItemSelectionModel.Select)
        # clear previous and apply new selection
        model.clearSelection()
        if len(rows) > 0:
            model.select(selection, QtCore.QItemSelectionModel.Select)
        model.blockSignals(False)


if __name__ == "__main__":
    pass
