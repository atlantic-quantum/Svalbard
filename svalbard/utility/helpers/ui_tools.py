"""Tools for handling QT UI intefaces."""

import sys
from typing import Callable

from qtpy.QtCore import QObject, QPoint, Qt
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import (
    QAction,
    QApplication,
    QFrame,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QWidget,
)

MAC = sys.platform == "darwin"
WIN = sys.platform.startswith("win")
LINUX = sys.platform.startswith("linux")


def is_dark_mode() -> bool:
    """Return True if dark mode is enabled on the system.

    Returns
    -------
    bool
        True if dark mode is enabled on the system.
    """
    # pylint: disable=import-outside-toplevel, import-error
    from qtpy.QtCore import QSettings

    # app = QApplication.instance()
    # return app.styleHints().colorScheme() == Qt.ColorScheme.Dark
    # pylint: enable=import-outside-toplevel, import-error
    settings = QSettings()
    if MAC:
        # check for dark mode on mac
        return settings.value("AppleInterfaceStyle") == "Dark"
    if WIN:
        # check for dark mode on windows
        app = QApplication.instance()
        return app.styleHints().colorScheme() == Qt.ColorScheme.Dark
        # return (
        #     settings.value(
        #         "HKEY_CURRENT_USER\\Software\\Microsoft\\Windows\\CurrentVersion\\Themes\\Personalize\\AppsUseLightTheme"  # noqa: E501
        #     )
        #     == 0
        # )
    if LINUX:
        # check for dark mode on linux
        return (
            settings.value("org.gnome.desktop.interface", "gtk-theme") == "Adwaita-dark"
        )
    # default to light mode
    return False


def get_theme_icon(name: str) -> str:
    """Return name of icon that reflects current theme (dark or light).

    Parameters
    ----------
    name : str
        Name of icon

    Returns
    -------
    str
        Name of icon with theme suffix
    """
    if is_dark_mode():
        name = name + "-dark"
    # add path to resource if not already there
    if not name.startswith(":/"):
        name = ":/" + name
    return name


def get_max_screen_height() -> int:
    """Get maximum screen height for all available screens.

    Returns
    -------
    int
        Maximum screen height
    """
    heights = [x.availableVirtualSize().height() for x in QApplication.screens()]
    return max(heights)


def force_coords_inside_screen(position: QPoint) -> QPoint:
    """Make sure coordinates are inside the available screen

    Parameters
    ----------
    position : QPoint
        Coordinate to force into screen

    Returns
    -------
    QPoint
        Coordinate guaranteed to be inside available screen real estate
    """
    try:
        # make sure coordinates are inside the available screen
        max_pos = QApplication.screens()[0].availableVirtualSize()
        # add some margin
        if position.x() > (max_pos.width() - 50):
            position.setX(10)
        if position.y() > (max_pos.height() - 50):
            position.setY(10)
        return position
    except Exception:
        # return (0, 0) for any exception
        return QPoint(0, 0)


def create_action(
    parent: QObject,
    text: str,
    slot: Callable = None,
    checkable=False,
    icon: str | QIcon = None,
    icon_text: str = None,
    tip: str = None,
    shortcut: str = None,
) -> QAction:
    """Create actions for adding to toolbars/menubars

    Parameters
    ----------
    parent : QObject
        Parent QObject owning action
    text : str
        Descriptive text
    slot : Callable, optional
        Function to call on action, by default None
    checkable : bool, optional
        Checkable on/off, by default False
    icon : str | QIcon, optional
        Icon, either as resource string or QIcon, by default None
    icon_text : str, optional
        Descriptive text next to icon, by default None
    tip : str, optional
        Extensive tooltip describing action, by default None
    shortcut : str, optional
        Keyboard shortcut, by default None

    Returns
    -------
    QAction
        QAction to be place in UI
    """
    action = QAction(text, parent)
    if icon is not None:
        action.setIcon(QIcon(icon))
    # set tooltip to same as text if not given
    if tip is None:
        tip = text
    action.setToolTip(tip)
    action.setStatusTip(tip)
    if shortcut is not None:
        action.setShortcut(shortcut)
    if slot is not None:
        action.triggered.connect(slot)
    if checkable:
        action.setCheckable(True)
    if icon_text is not None:
        action.setIconText(icon_text)
    return action


def add_actions(target: QObject, actions: list[QAction | None]) -> None:
    """Add a list of actions to a toolbar or menubar.

    Parameters
    ----------
    target : QObject
        UI element to add actions to.
    actions : list[QAction  |  None]
        List of actions to add. None gives a seperator
    """
    for action in actions:
        if action is None:
            # add separator if None
            separator = QAction(target)
            separator.setSeparator(True)
            target.addAction(separator)
        else:
            target.addAction(action)


def create_tool_button(
    text: str,
    slot: Callable = None,
    tip: str = None,
    shortcut: str = None,
    checkable=False,
    icon: str = None,
    small=False,
) -> QToolButton:
    """Create a tool button and connect callback to slot

    Parameters
    ----------
    text : str
        Button text
    slot : Callable, optional
        Function callback, by default None
    tip : str, optional
        Tootip, by default None
    shortcut : str, optional
        Keyboard shortcut, by default None
    checkable : bool, optional
        Checkable on/off, by default False
    icon : str, optional
        Icon, as resource string, by default None
    small : bool, optional
        Use small size, by default False

    Returns
    -------
    QToolButton
        Final button
    """
    # style different for mac
    button = QToolButton() if (MAC or small) else QPushButton()
    button.setText(text)
    if tip is not None:
        button.setToolTip(tip)
        button.setStatusTip(tip)
    if shortcut is not None:
        button.setShortcut(shortcut)
    if slot is not None:
        button.clicked.connect(slot)
    if checkable:
        button.setCheckable(True)
    if icon is not None:
        button.setIcon(QIcon(icon))
    return button


def create_line(vertical=False) -> QFrame:
    """reate a line element, vertical or horizontal

    Parameters
    ----------
    vertical : bool, optional
        If true, vertical, else horizontal, by default False

    Returns
    -------
    QFrame
        Line object
    """
    line = QFrame()
    if vertical:
        line.setGeometry(1, 1, 0, 100)
        line.setFrameShape(QFrame.VLine)
    else:
        line.setGeometry(1, 1, 100, 0)
        line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Sunken)
    return line


def create_spacer(size: int = None, vertical=False, minimal=False) -> QFrame:
    """Create a spacer QFrame, vertical or horizontal

    Parameters
    ----------
    size : int, optional
        Size in pixels, by default None
    vertical : bool, optional
        Orientation, vertical (True) or horizontal (False), by default False
    minimal : bool, optional
        Make as tight as possible, by default False

    Returns
    -------
    QFrame
        Spacer QFrame item
    """

    qs = QFrame()
    if size is None:
        # expand in all directions
        qs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    elif vertical:
        # vertical line
        if minimal:
            qs.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            qs.setFixedWidth(1)
            qs.setFixedHeight(size)
        else:
            qs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            qs.setFixedHeight(size)
    else:
        # horizontal line
        if minimal:
            qs.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            qs.setFixedWidth(size)
            qs.setFixedHeight(1)
        else:
            qs.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
            qs.setFixedWidth(size)
    return qs


def raise_window(window: QWidget) -> None:
    """Raise a dialog window to be top-most, including un-minimizing it

    Parameters
    ----------
    window : QWidget
        Window to raise
    """
    # update window state to not be hidden or minimized
    window.setWindowState(
        (window.windowState() & (~Qt.WindowMinimized)) | Qt.WindowActive
    )
    # show, raise and activate
    window.show()
    window.raise_()
    window.activateWindow()


def set_tip(widget: QWidget, text: str) -> None:
    """Set tip text, including tooltip, status bar and balloon pop-up

    Parameters
    ----------
    widget : QWidget
        UI element
    text : str
        Tip to add
    """
    widget.setToolTip(text)
    widget.setStatusTip(text)


if __name__ == "__main__":
    print(is_dark_mode())
    pass
