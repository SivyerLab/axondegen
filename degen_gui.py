"""
Test of gui
"""

import logging
import sys
import traceback as tb
from pathlib import Path

from image_process import ImageProcess

import numpy as np
import scipy as sp

import cv2
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore

__author__ = 'Alexander Tomlinson'
__email__ = 'tomlinsa@ohsu.edu'
__version__ = '0.0.1'

# setup logging
logger = logging.getLogger('gridcreator')
logger.setLevel(logging.INFO)

logger_handler = logging.StreamHandler(sys.stdout)
logger_handler.setLevel(logging.INFO)

logger.addHandler(logger_handler)


# setup exception handler:
# PyQt4 would print errors to IDE console and suppress them, but
# PyQt5 just terminates without printing a traceback

def except_hook(type, value, traceback, frame):
    """
    Catches and logs exceptions without crashing pyqt

    :param type:
    :param value:
    :param traceback:
    :param error_dialog:
    :return:
    """
    # logger.error(''.join(tb.format_exception(*(type, value, traceback))))
    frame.show_traceback(type, value, traceback)

    sys.__excepthook__(type, value, traceback)


class Frame(QtWidgets.QMainWindow):
    """
    Main window
    """
    def __init__(self, parent=None):
        """
        init
        """
        super(Frame, self).__init__(parent)

        # self.setGeometry(100, 100, 600, 500)
        self.setWindowTitle('CRC OCT-Overlay')

        self.menubar = self.create_menu_bar()
        self.statusbar = self.create_status_bar()

        self.error_dialog = QtWidgets.QErrorMessage()
        self.error_dialog.setFixedWidth(600)
        self.error_dialog.setMinimumHeight(300)

        # init central widget
        self.main_widget = CentralWidget(self)

        self.setCentralWidget(self.main_widget)

        self.show()

    def show_traceback(self, type, value, traceback):
        """
        Formatting error dialog
        """
        self.error_dialog.showMessage('<br><br>'.join(tb.format_exception(*(type, value, traceback))))

    def create_menu_bar(self):
        """
        Creates the menu bar
        """
        menubar = QtWidgets.QMenuBar()
        self.setMenuBar(menubar)
        return menubar

    def create_status_bar(self):
        """
        Creates the status bar
        """
        statusbar = QtWidgets.QStatusBar()
        statusbar.setSizeGripEnabled(False)

        self.setStatusBar(statusbar)
        statusbar.showMessage('Welcome', 2000)

        return statusbar

    def closeEvent(self, event):
        """
        Intercept close event to properly shut down camera and thread.
        :param event:
        """
        super(Frame, self).closeEvent(event)


class CentralWidget(QtWidgets.QWidget):
    """
    Main frame.
    """
    def __init__(self, parent=None):
        """
        init
        """
        super(CentralWidget, self).__init__(parent)
        self.frame = parent
        self.im = None
        self.im_process = None
        self.centers = None

        # setup status and menu bars
        self.setup_status_bar()
        self.setup_menu_bar()

        # setup tabs
        self.tabs = QtWidgets.QTabWidget()

        # viewer tab
        layout_viewer = self.setup_viewer()
        self.viewer_tab = QtWidgets.QWidget()
        self.tabs.addTab(self.viewer_tab, 'Viewer')
        self.viewer_tab.setLayout(layout_viewer)

        # selector tab
        # layout_selector = self.setup_selector()
        self.sel_tab = QtWidgets.QWidget()
        self.tabs.addTab(self.sel_tab, 'Select')
        # self.sel_tab.setLayout(layout_selector)

        # right side controls
        layout_controls = self.setup_controls()

        # top level layout
        layout_frame = QtWidgets.QHBoxLayout()

        layout_frame.addWidget(self.tabs)
        layout_frame.addLayout(layout_controls)

        self.setLayout(layout_frame)
        self.show()

    def setup_status_bar(self):
        """
        Sets up the labels in the status bar
        """
        self.status_mouse = QtWidgets.QLabel('x:{:4.0f} | y:{:4.0f} '.format(0, 0))
        self.status_count = QtWidgets.QLabel('Count: {} '.format(0))

        self.frame.statusbar.addPermanentWidget(self.status_count)
        self.frame.statusbar.addPermanentWidget(self.status_mouse)

    def setup_menu_bar(self):
        """
        Sets up the menus in the menu bar
        """
        menu_file = self.frame.menubar.addMenu('File')

        # symmetry sub menu
        # symmetry_menu = QtWidgets.QMenu('Symmetry', self)

        file_open = QtWidgets.QAction('Open', self, statusTip='Open image in viewer', shortcut='Ctrl+O')
        file_open.triggered.connect(self.on_menu_file_open)

        menu_file.addAction(file_open)

    def setup_viewer(self):
        """
        Sets up right pane with viewer

        :return: layout with viewer and related controls
        """
        viewer_splitter = QtWidgets.QVBoxLayout()

        self.checkbox_method = QtWidgets.QCheckBox('show method')
        self.checkbox_method.toggled.connect(self.on_checkbox_method)

        self.image_viewer = ImageViewer(self)

        self.image_viewer.scene().sigMouseMoved.connect(self.image_viewer.on_mouse_move)
        # self.image_viewer.scene().sigMouseClicked.connect(self.on_mouse_click)

        viewer_splitter.addWidget(self.image_viewer)

        button_splitter = QtWidgets.QHBoxLayout()

        button_splitter.addStretch()
        button_splitter.addWidget(self.checkbox_method)

        viewer_splitter.addLayout(button_splitter)

        return viewer_splitter

    def setup_controls(self):
        """
        Sets up the controls for the frame

        :return: layout with the controls
        """
        # TODO: factor out into separate class?

        # setup controls
        layout_control_splitter = QtWidgets.QVBoxLayout()

        self.checkbox_autolevel = QtWidgets.QCheckBox('check 1')
        # self.checkbox_autolevel.setChecked(True)
        layout_control_splitter.addWidget(self.checkbox_autolevel)
        # self.checkbox_autolevel.stateChanged.connect(self.on_checkbox_autolevel)

        self.checkbox_threshold = QtWidgets.QCheckBox('check 2')
        # self.checkbox_threshold.setChecked(True)
        layout_control_splitter.addWidget(self.checkbox_threshold)
        # self.checkbox_threshold.stateChanged.connect(self.on_checkbox_threshold)

        self.button_find_degen = QtWidgets.QPushButton('find degen')
        layout_control_splitter.addWidget(self.button_find_degen)
        self.button_find_degen.clicked.connect(self.on_button_find_degen)

        self.button_overlay = QtWidgets.QPushButton('button 2')
        layout_control_splitter.addWidget(self.button_overlay)
        # self.button_overlay.clicked.connect(self.on_button_overlay)

        layout_control_splitter.setAlignment(QtCore.Qt.AlignTop)

        return layout_control_splitter

    def on_menu_file_open(self):
        """
        Handles opening excel files
        """
        im_path = QtWidgets.QFileDialog.getOpenFileName(self, 'Open image',
                                                                    filter='TIF (*.tif);;'\
                                                                           'PNG (*.png);;'\
                                                                           'All files (*)')[0]
        if not im_path:
            return

        im_path = Path(im_path)
        assert im_path.exists()

        self.im = cv2.imread(str(im_path))

        self.image_viewer.set_im(self.im)

    def on_button_find_degen(self):
        """
        Handles getting degen centers

        :return:
        """
        self.im_process = ImageProcess(self.im)

        # get mask
        m = self.im_process.get_degen_dt()
        c = self.im_process.get_centroids(m, min_dist=20)

        self.centers = c

        self.status_count.setText('Count: {} '.format(len(c)))

        self.image_viewer.plot_centers(self.centers)

    def on_checkbox_method(self):
        """
        Handles switching to method view

        :return:
        """
        if self.checkbox_method.isChecked():
            if self.im_process is not None:
                if self.im_process.method_im is not None:
                    self.image_viewer.set_im(self.im_process.method_im, clear=False)
                    return

            self.checkbox_method.blockSignals(True)
            self.checkbox_method.setChecked(False)
            self.checkbox_method.blockSignals(False)

        else:
            self.image_viewer.set_im(self.im, clear=False)


class ImageViewer(pg.GraphicsLayoutWidget):

    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self.parent = parent

        # create plot for points and stuff
        self.plot = self.addPlot(0, 0)
        self.plot.hideAxis('left')
        self.plot.hideAxis('bottom')
        self.plot.invertY()

        self.viewer = pg.ImageItem(axisOrder='row-major')
        self.plot.addItem(self.viewer)
        self.plot.setAspectLocked(True)

        c = (255, 0, 0, 128)
        self.scatter = pg.ScatterPlotItem(pen=c, symbolBrush=c, symbolPen='w', symbol='+', symbolSize=14)
        # TODO: remove after done
        self.set_im()
        self.plot.addItem(self.scatter)

    def sizeHint(self):
        return QtCore.QSize(600, 600)

    def set_im(self, im=None, clear=True):
        """
        Updates the viewer to an image

        :return:
        """
        if im is None:
            im_path = r'\\Sivyer-rigb\d\raw_data\Confocal\681_grade 2_scale20um.tif'
            im = cv2.imread(str(Path(im_path)))
            self.parent.im = im

        if clear:
            self.scatter.clear()
            self.parent.status_count.setText('Count: {} '.format(0))

            self.parent.checkbox_method.blockSignals(True)
            self.parent.checkbox_method.setChecked(False)
            self.parent.checkbox_method.blockSignals(False)

        self.viewer.setImage(im)

        # TODO: figure out aesthetics here
        # x, y = im.shape[:2]
        # self.plot.setAspectLocked(True)
        # self.plot.setLimits(xMin=0,
        #                     xMax=self.viewer.width(),
                            # yMin=0,
                            # yMax=self.viewer.height()
                            # )
        # self.plot.setAspectLocked()
        # self.plot.setRange(xRange=[0,x], yRange=[0,y], padding=0)

    def on_mouse_move(self, pos):
        """
        Updates the mouse pos in status bar

        :param pos:
        :return:
        """
        im_coords = self.viewer.mapFromScene(pos)
        x, y = im_coords.x(), im_coords.y()
        self.parent.status_mouse.setText('x:{:4.0f} | y:{:4.0f} '.format(x, y))

    def plot_centers(self, centers):
        """
        Plots scatter plot of centroids

        :param centers:
        :return:
        """
        self.scatter.setData(pos=centers)


def main():
    """
    main function
    """
    app = QtWidgets.QApplication([])
    frame = Frame()

    # catch errors into error dialog
    sys.excepthook = lambda x, y, z: except_hook(x, y, z, frame)

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()