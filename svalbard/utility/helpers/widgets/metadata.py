"""Dialogs for user input and information"""

from qtpy import QtWidgets
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QAbstractItemView, QHeaderView

from ...data_model.instruments import InstrumentModel
from ...data_model.measurement.measurement import Measurement
from . import ui_controls, ui_tools
from .ui_tools import MAC


class SidebarTreeWidget(QtWidgets.QTreeWidget):
    """A tree widget for showing configuration settings in a sidebar

    Parameters
    ----------
    header : list, optional
        List of column headers, by default []
    """

    def __init__(self, header=[], parent=None):
        super().__init__(parent)
        # configure look of tree
        self.setColumnCount(len(header))
        self.setHeaderLabels(header)
        self.setHeaderHidden(True)
        # hide frame on windows
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        # make sure columns are resized to contents
        for n in range(len(header)):
            self.header().setSectionResizeMode(n, QHeaderView.ResizeToContents)
        self.setExpandsOnDoubleClick(True)
        self.setSortingEnabled(False)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.setUniformRowHeights(True)
        self.setItemsExpandable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # dark mode
        if ui_tools.is_dark_mode():
            self.setStyleSheet(
                "background-color: transparent;"
                + "selection-background-color: '#535353';"
                + "selection-color: white"
            )
        else:
            self.setStyleSheet("background-color: transparent")
        self.setIndentation(16)
        # adjust font sizes on mac os
        if MAC:
            qfont = self.font()
            qfont.setPointSize(12)
            self.setFont(qfont)
            header = self.headerItem()
            for n in range(header.columnCount()):
                font = header.font(n)
                font.setPointSize(11)
                header.setFont(n, font)


class ExpandingSidebarTreeWidget(SidebarTreeWidget):
    """An auto expanding/collapsing widget for showing settings in a sidebar

    Parameters
    ----------
    header : list, optional
        List of column headers, by default []
    """

    def __init__(self, header=[], parent=None):
        super().__init__(header=header, parent=parent)
        # disable vertical scroll bar - widget will expand to fit content
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # connect signals to update height on expand/collapse
        self.expanded.connect(self._update_height)
        self.collapsed.connect(self._update_height)

    def _update_height(self):
        """Update height of widget to fit content"""
        self.setMaximumHeight(self.sizeHint().height())

    def _get_height(self, parent=None) -> int:
        """Get height of item in tree widget

        Parameters
        ----------
        parent : QTreeWidgetItem, optional
            Parent item, by default None

        Returns
        -------
        int
            Height of item
        """
        height = 0
        if not parent:
            parent = self.rootIndex()
        for row in range(self.model().rowCount(parent)):
            child = self.model().index(row, 0, parent)
            height += self.rowHeight(child)
            if self.isExpanded(child) and self.model().hasChildren(child):
                height += self._get_height(child)
        return height

    def sizeHint(self):
        # overload sizeHint to return height of tree widget
        hint = super().sizeHint()
        hint.setHeight(self._get_height() + self.frameWidth() * 2 + 2)
        return hint

    def minimumSizeHint(self):
        # overload minimumSizeHint to return height of tree widget
        hint = super().minimumSizeHint()
        # prevent widget from expanding to larger than screen
        # hint.setHeight(min(self.sizeHint().height(), get_max_screen_height()-100))
        hint.setHeight(self.sizeHint().height())
        return hint


class InstrumentsTree(SidebarTreeWidget):
    """A tree widget for displaying instrument configurations"""

    def __init__(self, instruments: dict[str, InstrumentModel] = {}, parent=None):
        super().__init__(header=["Name", "Value"], parent=parent)
        self._instruments = instruments
        # create GUI and populate tree
        self.populate_tree()

    def set_instruments(self, instruments: dict[str, InstrumentModel] = {}):
        """Update instruments to show

        Parameters
        ----------
        instruments : dict[str, InstrumentModel]
            dict with instrument configurations
        """
        self._instruments = instruments

    def populate_tree(self, text: str = None):
        """Populate tree with instrument configuration

        Parameters
        ----------
        text : str, optional
            Filter string for setting to show, by default None
        """
        # filter is not case sensitive
        if text is not None:
            text = text.lower()
        # remember if tree is expanded
        expanded = []
        for n in range(self.topLevelItemCount()):
            if self.topLevelItem(n).isExpanded():
                expanded.append(self.topLevelItem(n).text(0))
        # clear old view
        self.clear()
        for _, instrument in self._instruments.items():
            # create top level item
            item = QtWidgets.QTreeWidgetItem([instrument.get_id_string(), ""])
            # disable selection of top-level items
            item.setFlags(item.flags() & ~Qt.ItemIsSelectable)
            # bold fonts
            qfont = item.font(0)
            qfont.setBold(True)
            item.setFont(0, qfont)
            # add children
            for key, setting in instrument.settings.items():
                # filter by text
                if text is not None and text not in key.lower():
                    continue
                child = QtWidgets.QTreeWidgetItem([key, setting.get_value_string()])
                item.addChild(child)
            # add top level item to tree view, unless it has no children
            if item.childCount() > 0:
                self.addTopLevelItem(item)
        # restore expanded state
        for n in range(self.topLevelItemCount()):
            # always expand if nothing was previously expanded
            if len(expanded) == 0 or self.topLevelItem(n).text(0) in expanded:
                self.topLevelItem(n).setExpanded(True)


class InstrumentsView(QtWidgets.QWidget):
    """A widget for displaying instrument settings, including search field"""

    def __init__(self, instruments: dict[str, InstrumentModel] = {}, parent=None):
        super().__init__(parent)
        self._instruments = instruments
        # create GUI and populate tree
        self._create_ui()

    def _create_ui(self):
        """Generate UI elements"""
        # create search field
        self.search_field = ui_controls.StringControl()
        self.search_field.setClearButtonEnabled(True)
        self.search_field.setPlaceholderText("Search...")
        self.search_field.textChanged.connect(self._on_search_changed)
        # tree view for instruments
        self.tree_instruments = InstrumentsTree(self._instruments)
        # combine search field and tag tree
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(QtWidgets.QLabel("Search instruments settings: "))
        layout.addWidget(self.search_field)
        layout.addSpacing(8)
        layout.addWidget(self.tree_instruments)
        self.setLayout(layout)

    def _on_search_changed(self, text: str):
        """Callback on tag selection change

        Parameters
        ----------
        text : str
            Current text in search field
        """
        # populate tree with filtered settings
        self.tree_instruments.populate_tree(text)

    def set_instruments(self, instruments: dict[str, InstrumentModel] = {}):
        """Update instruments to show

        Parameters
        ----------
        instruments : dict[str, InstrumentModel]
            dict with instrument configurations
        """
        self._instruments = instruments
        self.tree_instruments.set_instruments(instruments)
        # keep search field text
        self.tree_instruments.populate_tree(self.search_field.text())


class MeasurementView(QtWidgets.QWidget):
    """A widget for displaying the measurement step/log/relations configuration"""

    def __init__(self, measurement: Measurement = None, parent=None):
        super().__init__(parent)
        self._measurement = measurement
        # create UI
        self._create_ui()

    def _create_ui(self):
        """Generate UI elements"""
        # create step/log/relations views
        self.step_view = StepChannelView(self._measurement)
        self.log_view = LogChannelView(self._measurement)
        self.relations_view = RelationsView(self._measurement)
        # combine search field and tag tree
        layout = QtWidgets.QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addSpacing(4)
        layout.addWidget(self.step_view)
        layout.addWidget(self.log_view)
        layout.addWidget(self.relations_view)
        self.setLayout(layout)

    def set_measurement(self, measurement: Measurement):
        """Update measurement configuration

        Parameters
        ----------
        measurement : Measurement
            Measurement object
        """
        self._measurement = measurement
        self.step_view.set_measurement(measurement)
        self.log_view.set_measurement(measurement)
        self.relations_view.set_measurement(measurement)


class _MeasureViewItem(ExpandingSidebarTreeWidget):
    """Base class for tree widget for displaying the measurement configuration"""

    HEADER = []

    def __init__(self, measurement: Measurement, parent=None):
        super().__init__(header=self.HEADER, parent=parent)
        # store measurement configuration
        self._measurement = measurement
        # configure UI and populate tree
        self.setSelectionMode(QAbstractItemView.NoSelection)
        self.setRootIsDecorated(False)
        self.setIndentation(16)
        if self._measurement is not None:
            self.populate_tree()

    def set_measurement(self, measurement: Measurement):
        """Update measurement configuration

        Parameters
        ----------
        measurement : Measurement
            Measurement object
        """
        self._measurement = measurement
        if self._measurement is None:
            self.clear()
        else:
            self.populate_tree()

    def populate_tree(self):
        """Function for population tree with configuration, to be suclassed"""
        pass

    def _addItem(self, item: QtWidgets.QTreeWidgetItem):
        """Add item to tree, unless it has no children"""
        if item.childCount() > 0:
            self.addTopLevelItem(item)
            qfont = item.font(0)
            qfont.setBold(True)
            item.setFont(0, qfont)
            item.setFlags(item.flags() & ~Qt.ItemIsSelectable)


class StepChannelView(_MeasureViewItem):
    """A tree widget for displaying the step channel configuration"""

    HEADER = ["Name", "Values", "Step"]

    def populate_tree(self):
        """Populate tree with measurement configuration"""
        # clear old view
        self.clear()
        # create top level item
        step_items = QtWidgets.QTreeWidgetItem(["Step channels", ""])
        # add step item children
        for step_item in self._measurement.step_items:
            channel = self._measurement.get_channel(step_item.name)
            (range_str, step_str) = step_item.get_range_strings(channel)
            child = QtWidgets.QTreeWidgetItem(
                [f"{step_item.name}  ", f"{range_str}  ", step_str]
            )
            step_items.addChild(child)
        self._addItem(step_items)
        # always expand step and log items, collapse relations if there are too many
        step_items.setExpanded(True)


class LogChannelView(_MeasureViewItem):
    """A tree widget for displaying the step channel configuration"""

    HEADER = ["Name"]

    def populate_tree(self):
        """Populate tree with measurement configuration"""
        # clear old view
        self.clear()
        # log items
        log_items = QtWidgets.QTreeWidgetItem(["Log channels", ""])
        for ch in self._measurement.log_channels:
            child = QtWidgets.QTreeWidgetItem([ch])
            log_items.addChild(child)
        self._addItem(log_items)
        # always expand step and log items, collapse relations if there are too many
        log_items.setExpanded(True)


class RelationsView(_MeasureViewItem):
    """A tree widget for displaying the step channel configuration"""

    HEADER = ["Name", "Equation"]

    def populate_tree(self):
        """Populate tree with measurement configuration"""
        # clear old view
        self.clear()
        # relations
        relations = QtWidgets.QTreeWidgetItem(["Relations", ""])
        for relation in self._measurement.relations:
            child = QtWidgets.QTreeWidgetItem(
                [f"{relation.name}:  ", relation.equation]
            )
            relations.addChild(child)
        self._addItem(relations)
        # always expand step and log items, collapse relations if there are too many
        relations.setExpanded(True)


class TagSelector(ExpandingSidebarTreeWidget):
    """A tree widget for selecting tags to view"""

    def __init__(self, base_tag=(), parent=None):
        super().__init__(parent=parent)
        # init variables
        self.tags = {}
        self.base_tag = base_tag
        # the mapping is used to map tree entries (tuple of strings) to any data
        self.mapping = {}
        # configure GUI and populate tree
        self.configure_ui()
        self.populate_tree()

    def configure_ui(self):
        """Format tree to look correct"""
        title = "Tags" if len(self.base_tag) == 0 else self.base_tag[-1]
        header = [title]
        self.setColumnCount(1)
        self.setHeaderLabels(header)
        # set column widths
        self.setColumnWidth(0, 100)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)

    def populate_tree(self) -> None:
        """Clear and re-populate the tree view with Tags"""
        # clear old view
        self.clear()
        # create lookup mappings in both directions
        self._tree_items = dict()
        self._tree_tags = dict()
        # recursively add entries from dict
        self._add_tree_item(self.tags, self)

    def _add_tree_item(self, branch: dict, parent: QtWidgets.QTreeWidgetItem) -> None:
        """Recursively add tree items from a dict representing a branch

        Parameters
        ----------
        branch : dict
            Branch of tree with tags to add to UI
        parent : QtWidgets.QTreeWidgetItem
            Parent of branch
        """
        for key in branch:
            # create
            tag = self._tree_items.get(parent, ()) + (str(key),)
            item = QtWidgets.QTreeWidgetItem(parent, [key])
            # set font to bold for top-level items
            if isinstance(parent, TagSelector):
                qfont = item.font(0)
                qfont.setBold(True)
                item.setFont(0, qfont)
                # disable selection of top-level items
                item.setFlags(item.flags() & ~Qt.ItemIsSelectable)
            self._tree_items[item] = tag
            self._tree_tags[tag] = item
            if branch[key] is not None and isinstance(branch[key], dict):
                # recursively call to create child items
                self._add_tree_item(branch[key], item)

    def update_tags(self, tags: dict = {}, mapping: dict = {}) -> None:
        """Update available tags

        Parameters
        ----------
        tags : dict, optional
            Dictionary with tree of tags, by default {}
        mapping : dict, optional
            Mapping from list of tuples representing path to any value, by default {}
        """
        self.tags = tags
        self.mapping = mapping
        # block all signals while updating
        self.blockSignals(True)
        # get current selection
        selected = self.get_selected_tags()
        # populate tree
        self.populate_tree()
        # re-select items
        self.set_selected_tags(selected)
        self.expandAll()
        # re-enable signals
        self.blockSignals(False)

    def set_selected_tags(self, selected: list[tuple]) -> None:
        """Set selected tags

        Parameters
        ----------
        selected : list[tuple]
            Tags to select, as list of tuples representing the path
        """
        # block all signals while selecting
        self.blockSignals(True)
        # deselect old items
        self.clearSelection()
        # select items
        for tag in selected:
            if tag in self._tree_tags:
                item = self._tree_tags[tag]
                self.setCurrentItem(item)
        # re-enable signals
        self.blockSignals(False)

    def get_selected_tags(self) -> list[tuple]:
        """Return list of selected tags

        Returns
        -------
        list[tuple]
            List of selected tags, each tuple representing a path
        """
        return [self._tree_items[item] for item in self.selectedItems()]

    def get_selected_mapping(self) -> list[str]:
        """Return list of mapping of selected items, including children

        Returns
        -------
        list[str]
            List of selected items, using mapping provided when populationg the tree
        """

        # define recursive function to find all child item
        def selected_recursive(item, mapped_items):
            tag = self._tree_items[item]
            if tag in self.mapping:
                mapped_items.extend(self.mapping[tag])
            for n in range(item.childCount()):
                selected_recursive(item.child(n), mapped_items)
            return mapped_items

        # get mapping for item and all children using recursive function
        mapped_items = []
        for item in self.selectedItems():
            mapped_items = selected_recursive(item, mapped_items)
        return mapped_items
