"""Dialogs for user input and information"""

import os

from qtpy.QtCore import QObject, QPoint, QSettings, QSize, Qt, QTimer, Signal
from qtpy.QtGui import QKeySequence, QTextCursor
from qtpy.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QShortcut,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from . import ui_tools, ui_widgets


def show_message_box(
    message: str,
    detailed_message: str = None,
    title="Polaris",
    wait_for_response=False,
    parent: QObject = None,
):
    """Show message with optional detailed information

    Parameters
    ----------
    message : str
        Message to show
    title : str, optional
        Title of window, by default "Polaris"
    detailed_message : str, optional
        Detailed information, by default None
    wait_for_response : bool, optional
        If True, will halt program execution until closed, by default False
    parent : QObject, optional
        Parent QObject, by default None
    """
    message_box = QMessageBox(parent=parent)
    message_box.setWindowTitle(title)
    message_box.setText(message)
    if detailed_message is not None:
        message_box.setInformativeText(detailed_message)
    message_box.show()
    message_box.raise_()
    if wait_for_response:
        message_box.exec_()


class GetValueDialog(QDialog):
    """Dialog for getting numeric value from user.

    Parameters
    ----------
    value : _type_, optional
        _description_, by default None
    text : str, optional
        User prompt, by default 'Enter value:'
    title : str, optional
        Window title, by default 'Provide input'
    parent : _type_, optional
        _description_, by default None
    """

    SignalClose = Signal(object)

    def __init__(
        self,
        value=0.0,
        text="Enter value:",
        title="Provide input",
        parent: QWidget = None,
    ):
        super().__init__(parent)
        # label
        self.label_text = QLabel(text)
        # control
        self.value_widget = ui_widgets.FloatSpinBox()
        if value is not None:
            self.value_widget.setValue(value)
        # cancel button
        button_box = QDialogButtonBox(QDialogButtonBox.Ok)
        button_box.accepted.connect(self.close)
        # create layout for buttons
        layout_horizontal = QHBoxLayout()
        layout_horizontal.setContentsMargins(8, 8, 8, 8)
        layout_horizontal.setSpacing(0)
        layout_horizontal.addWidget(self.label_text)
        layout_horizontal.addWidget(self.value_widget)
        # create group
        self.group_value = QGroupBox(title)
        self.group_value.setLayout(layout_horizontal)
        # create layout
        layout_vertical = QVBoxLayout()
        layout_vertical.setContentsMargins(6, 6, 6, 4)
        layout_vertical.setSpacing(2)
        layout_vertical.addWidget(self.group_value)
        layout_vertical.addWidget(button_box)
        self.setLayout(layout_vertical)
        # set values
        self.update_dialog(value, text, title)
        # set window title
        self.setWindowTitle(title)
        # set close shortcut
        QShortcut(QKeySequence("Ctrl+W"), self, self.close)
        # resize and reposition
        self.setModal(True)
        self.setMaximumHeight(20)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self.resize(240, 50)
        self.setSizeGripEnabled(True)

    def update_dialog(
        self, value: float | None = None, text="Enter value:", title="User input"
    ):
        """Update dialog with new value and descriptions.

        Parameters
        ----------
        value : _type_, optional
            New value, by default None
        text : str, optional
            New prompt, by default 'Enter value:'
        title : str, optional
            New window title, by default 'User input'
        """
        self.label_text.setText(text)
        self.group_value.setTitle(title)
        if value is not None:
            self.value_widget.setValue(value)
            self.value_widget.selectAll()

    def closeEvent(self, event):
        # emit signal upon exiting
        self.SignalClose.emit(self.value_widget.value())
        event.accept()

    def reject(self):
        # emig signal upon exiting
        self.SignalClose.emit(self.value_widget.value())
        QDialog.reject(self)


class LoggingViewer(QDialog):
    """A viewer for the Python logging module, reading log data from a text file.

    Parameters
    ----------
    log_path : str
        Path to log file.
    refresh_interval : float, optional
        Refresh interval in seconds, by default 5.0
    title : str, optional
        Window title, by default 'Logging viewer'
    wrap_lines : bool, optional
        Wrap text lines, by default True
    qt_settings_name : str, optional
        Name of QT settings object, by default None
    parent : _type_, optional
        Parent widget, by default None
    """

    def __init__(
        self,
        log_path: str,
        refresh_interval=5.0,
        title="Logging viewer",
        wrap_lines=True,
        qt_settings_name: str = None,
        parent=None,
    ):
        super().__init__(parent)
        # store settings
        self.log_path = log_path
        self.refresh_interval = refresh_interval
        self.window_size = (600, 300)
        self.qt_settings_name = qt_settings_name
        # current position in log file
        self.log_position = 0
        # create UI, log text display
        self.text_edit = QPlainTextEdit()
        self.text_edit.clear()
        self.text_edit.setReadOnly(True)
        # wrap/no wrap checkbox
        self.check_wrap = QCheckBox("Wrap text")
        self.check_wrap.setChecked(wrap_lines)
        self.check_wrap.stateChanged.connect(self.set_wrap_mode)
        # clear view pushbutton
        clear_view = QPushButton("Clear view")
        clear_view.setFocusPolicy(Qt.NoFocus)
        clear_view.clicked.connect(self.clear_log_text)
        # close button
        button_box = QDialogButtonBox(QDialogButtonBox.Close)
        button = button_box.button(QDialogButtonBox.Close)
        button.setFocusPolicy(Qt.NoFocus)
        button_box.rejected.connect(self.reject)
        # create layout for buttons
        layout_horizontal = QHBoxLayout()
        layout_horizontal.setContentsMargins(0, 0, 0, 0)
        layout_horizontal.setSpacing(0)
        layout_horizontal.addWidget(clear_view)
        layout_horizontal.addSpacing(6)
        layout_horizontal.addWidget(self.check_wrap)
        layout_horizontal.addStretch()
        layout_horizontal.addWidget(button_box)
        # create layout
        layout_vertical = QVBoxLayout()
        layout_vertical.setContentsMargins(6, 6, 6, 2)
        layout_vertical.setSpacing(2)
        layout_vertical.addWidget(self.text_edit)
        layout_vertical.addLayout(layout_horizontal)
        self.setLayout(layout_vertical)
        # set window title
        self.setWindowTitle(title)
        # set close shortcut
        QShortcut(QKeySequence("Ctrl+W"), self, self.close)
        # resize and reposition
        self.setSizeGripEnabled(True)
        if self.qt_settings_name is not None:
            settings = QSettings()
            size = settings.value(
                f"{self.qt_settings_name}/Size",
                QSize(self.window_size[0], self.window_size[1]),
            )
            position = settings.value(
                f"{self.qt_settings_name}/Position", QPoint(200, 100)
            )
            position = ui_tools.force_coords_inside_screen(position)
            self.resize(size)
            self.move(position)
        # update GUI
        self.set_wrap_mode()
        # load log contents for the first time
        self.reload_log(first_call=True)

    def set_wrap_mode(self):
        """Set log text wrap mode on/off"""
        if self.check_wrap.isChecked():
            self.text_edit.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        else:
            self.text_edit.setLineWrapMode(QPlainTextEdit.NoWrap)

    def clear_log_text(self):
        """Clear the log view text"""
        self.text_edit.clear()

    def reload_log(self, first_call=False):
        """Reload log from text file"""
        try:
            if not os.path.exists(self.log_path):
                return
            # load new data
            f = open(self.log_path, mode="r", encoding="utf-8")
            # check if file has been flushed
            if self.log_position > os.path.getsize(self.log_path):
                # file cleared, reload file from start, clear old view
                self.log_position = 0
                self.text_edit.clear()
            # only get data from old log position and onwards
            f.seek(self.log_position)
            data = f.read()
            # get new position
            self.log_position = f.tell()
            # close file
            f.close()
            # add data to log display
            if len(data) > 0:
                self.text_edit.moveCursor(QTextCursor.End)
                self.text_edit.insertPlainText(data)
        finally:
            # call again in a few seconds if window is shown
            if first_call or self.isVisible():
                QTimer.singleShot(int(1000 * self.refresh_interval), self.reload_log)

    def closeEvent(self, event=None):
        # save window settings
        if self.qt_settings_name is not None:
            settings = QSettings()
            settings.setValue(f"{self.qt_settings_name}/Size", self.size())
            settings.setValue(f"{self.qt_settings_name}/Position", self.pos())
        # close
        event.accept()

    def sizeHint(self):
        return QSize(self.window_size[0], self.window_size[1])


if __name__ == "__main__":
    pass
