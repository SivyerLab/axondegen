"""
Test of gui
"""

import logging
import sys
import traceback as tb
from pathlib import Path

import cv2
import numpy as np
import scipy as sp
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore, QtGui

from image_process import ImageProcess

__author__ = 'Alexander Tomlinson'
__email__ = 'tomlinsa@ohsu.edu'
__version__ = '0.0.1'

# setup logging
logger = logging.getLogger('AxonDegen')
logger.setLevel(logging.INFO)

logger_handler = logging.StreamHandler(sys.stdout)
logger_handler.setLevel(logging.INFO)

logger.addHandler(logger_handler)


# setup exception handler:
# PyQt4 would print errors to console and suppress them, but
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
        self.setWindowTitle('Axon Degeneration')

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

        self.current_view = 'im'

        self.thresh_value = 125
        self.fx_value = 6

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
        layout_selector = self.setup_selector()
        self.sel_tab = QtWidgets.QWidget()
        self.tabs.addTab(self.sel_tab, 'Select')
        self.sel_tab.setLayout(layout_selector)

        self.image_viewer.set_im(None)

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

        file_open = QtWidgets.QAction('Open', self, statusTip='Open image in viewer', shortcut='Ctrl+O')
        file_open.triggered.connect(self.on_menu_file_open)

        menu_file.addAction(file_open)

    def setup_viewer(self):
        """
        Sets up right pane with viewer

        :return: layout with viewer and related controls
        """
        viewer_splitter = QtWidgets.QVBoxLayout()

        self.checkbox_binary = QtWidgets.QCheckBox('show binary')
        self.checkbox_binary.toggled.connect(self.on_checkbox_binary)

        self.checkbox_method = QtWidgets.QCheckBox('show method')
        self.checkbox_method.toggled.connect(self.on_checkbox_method)

        self.checkbox_result = QtWidgets.QCheckBox('show result')
        self.checkbox_result.toggled.connect(self.on_checkbox_result)

        self.image_viewer = GenericViewer(self)
        # hackish bad fix
        self.image_viewer.sizeHint = lambda: QtCore.QSize(600, 600)

        self.image_viewer.scene().sigMouseMoved.connect(self.image_viewer.on_mouse_move)
        # self.image_viewer.scene().sigMouseClicked.connect(self.on_mouse_click)

        viewer_splitter.addWidget(self.image_viewer)

        button_splitter = QtWidgets.QHBoxLayout()

        button_splitter.addStretch()
        button_splitter.addWidget(self.checkbox_binary)
        button_splitter.addWidget(self.checkbox_method)
        button_splitter.addWidget(self.checkbox_result)

        viewer_splitter.addLayout(button_splitter)

        return viewer_splitter

    def setup_controls(self):
        """
        Sets up the controls for the frame

        :return: layout with the controls
        """
        # TODO: factor out into separate class?

        # splitter to return
        layout_control_splitter = QtWidgets.QVBoxLayout()

        # self.checkbox_autolevel = QtWidgets.QCheckBox('check 1')
        # self.checkbox_autolevel.setChecked(True)
        # layout_control_splitter.addWidget(self.checkbox_autolevel)
        # self.checkbox_autolevel.stateChanged.connect(self.on_checkbox_autolevel)

        # self.checkbox_threshold = QtWidgets.QCheckBox('check 2')
        # self.checkbox_threshold.setChecked(True)
        # layout_control_splitter.addWidget(self.checkbox_threshold)
        # self.checkbox_threshold.stateChanged.connect(self.on_checkbox_threshold)

        self.button_find_degen = QtWidgets.QPushButton('find degen')
        layout_control_splitter.addWidget(self.button_find_degen)
        self.button_find_degen.clicked.connect(self.on_button_find_degen)

        self.button_overlay = QtWidgets.QPushButton('button 2')
        layout_control_splitter.addWidget(self.button_overlay)
        # self.button_overlay.clicked.connect(self.on_button_overlay)

        layout_slider_threshold = self.setup_slider_threshold()

        layout_sliders = QtWidgets.QHBoxLayout()
        layout_sliders.addLayout(layout_slider_threshold)

        layout_control_splitter.addLayout(layout_sliders)

        layout_control_splitter.setAlignment(QtCore.Qt.AlignTop)

        return layout_control_splitter

    def setup_slider_threshold(self):
        """
        Sets up the slider for opacity
        :return: layout with slider and opacity
        """
        self.slider_threshold = QtGui.QSlider(QtCore.Qt.Vertical)
        self.slider_threshold.setMinimum(0)
        self.slider_threshold.setMaximum(255)
        self.slider_threshold.setValue(self.thresh_value)
        self.slider_threshold.setTickPosition(QtGui.QSlider.TicksRight)
        self.slider_threshold.setTickInterval(64)
        self.slider_threshold.valueChanged.connect(self.on_slider_threshold)

        value = self.thresh_value
        self.label_slider_threshold = QtGui.QLabel('{}'.format(value))

        layout_slider_label = QtGui.QVBoxLayout()
        layout_slider_label.addWidget(self.slider_threshold)
        layout_slider_label.addWidget(self.label_slider_threshold)

        return layout_slider_label

    def setup_selector(self):
        """
        Sets up the selector UI for manually highlighting degen axons

        :return:
        """
        # splitter to return
        layout_selector_splitter = QtWidgets.QHBoxLayout()

        # left panel with overview, controls, and cell views
        layout_left_panel = QtWidgets.QVBoxLayout()

        max_width = 341

        # overview
        self.overview = OverViewer(self)

        self.overview.setMinimumWidth(241)
        self.overview.setMaximumHeight(300)
        layout_left_panel.addWidget(self.overview)

        self.overview.scene().sigMouseClicked.connect(self.overview.on_mouse_click)

        # left panel mid controls
        layout_left_controls = QtWidgets.QVBoxLayout()

        self.button_done = QtWidgets.QPushButton('Done')
        self.button_done.clicked.connect(self.overview.on_button_done)

        self.degen_count = QtWidgets.QLabel('Count: {} '.format(0))

        layout_left_controls.addWidget(self.button_done)
        layout_left_controls.addWidget(self.degen_count)
        layout_left_controls.setAlignment(QtCore.Qt.AlignLeft)

        layout_left_panel.addLayout(layout_left_controls)

        # grid of selected cells
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        self.selected = SelectedViewer(self)
        selected_layout = self.selected.get_layout()

        layout_grid = QtWidgets.QGridLayout()

        scroll_widget = QtWidgets.QWidget()
        scroll_widget.setLayout(selected_layout)

        scroll_area.setWidget(scroll_widget)

        layout_grid.setAlignment(QtCore.Qt.AlignLeft)

        layout_left_panel.addWidget(scroll_area)

        # right side cell selector
        self.selector = SelectorViewer(self)

        layout_selector_splitter.addLayout(layout_left_panel, 1)
        layout_selector_splitter.addWidget(self.selector, 4)

        return layout_selector_splitter

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
        Handles getting degenerated centers

        :return:
        """
        self.im_process = ImageProcess(self.im)

        # get mask
        m = self.im_process.get_degen_mask('dt', thresh=self.thresh_value)
        c = self.im_process.get_centroids(m, min_dist=30)

        self.centers = c

        self.status_count.setText('Count: {} '.format(len(c)))

        if self.current_view == 'binary':
            self.image_viewer.set_im(self.im_process.thresh)
        elif self.current_view == 'method':
            self.image_viewer.set_im(self.im_process.method)
        elif self.current_view == 'result':
            self.image_viewer.set_im(self.im_process.result)

        self.image_viewer.plot_centers(self.centers)

    def on_checkbox_method(self):
        """
        Handles switching to method view

        :return:
        """
        if self.checkbox_method.isChecked():
            if self.im_process is not None:
                if self.im_process.method is not None:
                    self.image_viewer.set_im(self.im_process.method, clear=False)

                    self.checkbox_binary.blockSignals(True)
                    self.checkbox_binary.setChecked(False)
                    self.checkbox_binary.blockSignals(False)

                    self.checkbox_result.blockSignals(True)
                    self.checkbox_result.setChecked(False)
                    self.checkbox_result.blockSignals(False)

                    self.current_view = 'method'

                    return

            self.checkbox_method.blockSignals(True)
            self.checkbox_method.setChecked(False)
            self.checkbox_method.blockSignals(False)

        else:
            self.image_viewer.set_im(self.im, clear=False)
            self.current_view = 'im'

    def on_checkbox_binary(self):
        """
        Handles switching to method view

        :return:
        """
        if self.checkbox_binary.isChecked():
            if self.im_process is not None:
                if self.im_process.thresh is not None:
                    self.image_viewer.set_im(self.im_process.thresh, clear=False)

                    self.checkbox_method.blockSignals(True)
                    self.checkbox_method.setChecked(False)
                    self.checkbox_method.blockSignals(False)

                    self.checkbox_result.blockSignals(True)
                    self.checkbox_result.setChecked(False)
                    self.checkbox_result.blockSignals(False)

                    self.current_view = 'binary'

                    return

            self.checkbox_binary.blockSignals(True)
            self.checkbox_binary.setChecked(False)
            self.checkbox_binary.blockSignals(False)

        else:
            self.image_viewer.set_im(self.im, clear=False)
            self.current_view = 'im'

    def on_checkbox_result(self):
        """
        Handles switching to result view

        :return:
        """
        if self.checkbox_result.isChecked():
            if self.im_process is not None:
                if self.im_process.result is not None:
                    self.image_viewer.set_im(self.im_process.result, clear=False)

                    self.checkbox_binary.blockSignals(True)
                    self.checkbox_binary.setChecked(False)
                    self.checkbox_binary.blockSignals(False)

                    self.checkbox_method.blockSignals(True)
                    self.checkbox_method.setChecked(False)
                    self.checkbox_method.blockSignals(False)

                    self.current_view = 'result'

                    return

            self.checkbox_result.blockSignals(True)
            self.checkbox_result.setChecked(False)
            self.checkbox_result.blockSignals(False)

        else:
            self.image_viewer.set_im(self.im, clear=False)
            self.current_view = 'im'

    def on_slider_threshold(self):
        """
        Changes the opacity of the overlay
        """
        value = self.slider_threshold.value()
        self.thresh_value = value

        self.label_slider_threshold.setText('{}'.format(value))

        self.on_button_find_degen()


class ImageWidget(pg.GraphicsLayoutWidget):

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

        # TODO: move scatter to right spot
        c = (255, 0, 0, 128)
        self.scatter = pg.ScatterPlotItem(pen=c, symbolBrush=c, symbolPen='w', symbol='+', symbolSize=14)
        self.plot.addItem(self.scatter)

    def set_im(self, im=None, clear=True):
        """
        Updates the viewer to an image

        :return:
        """
        self.viewer.setImage(im)

        # TODO: figure out aesthetics here
        # x, y = im.shape[:2]
        # self.plot.setLimits(xMin=0,
        #                     xMax=self.viewer.width(),
        #                     yMin=0,
        #                     yMax=self.viewer.height()
        #                     )
        # self.plot.setAspectLocked()
        # self.plot.setRange(xRange=[0,x], yRange=[0,y], padding=0)

    def plot_centers(self, centers, clear=True):
        """
        Plots scatter plot of centroids

        :param centers:
        :return:
        """
        if clear:
            self.scatter.clear()

        if len(centers):
            self.scatter.setData(pos=centers)


class GenericViewer(ImageWidget):

    def __init__(self, parent=None):
        super().__init__(parent=parent)

    def set_im(self, im=None, clear=True):
        """
        Updates the viewer to an image

        :return:
        """
        if im is None:
            im_path = r'C:\Users\Alex\Downloads\morrison_slides\tifs\slide01\raw\slide01_section1_area08.tif'
            im_path = Path(im_path)
            assert im_path.exists()
            im = cv2.imread(str(im_path))
            self.parent.im = im

        if clear:
            self.scatter.clear()
            self.parent.status_count.setText('Count: {} '.format(0))

            self.parent.checkbox_method.blockSignals(True)
            self.parent.checkbox_method.setChecked(False)
            self.parent.checkbox_method.blockSignals(False)

        super().set_im(im, clear)

        # TODO: make this not reset grids if not new image
        self.parent.overview.grid_coord = (0, 0)
        self.parent.overview.set_im()
        self.parent.selector.set_im()

    def on_mouse_move(self, pos):
        """
        Updates the mouse pos in status bar

        :param pos:
        :return:
        """
        im_coords = self.viewer.mapFromScene(pos)
        x, y = im_coords.x(), im_coords.y()
        self.parent.status_mouse.setText('x:{:4.0f} | y:{:4.0f} '.format(x, y))


class OverViewer(GenericViewer):

    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self.grid_coord = (0, 0)
        self.spacing = 550

        self.grid = None
        self.grid_rects = None
        self.grid_fills = None

    def set_im(self, im=None, clear=True):
        """
        Updates the viewer to an image

        :return:
        """
        self.viewer.setImage(self.parent.im)
        self.draw_grid()

        self.grid_flags = np.full((self.grid.num_y, self.grid.num_x), fill_value=False, dtype=bool)
        # self.selected = np.array((self.grid.num_y, self.grid.num_x, 1), dtype=list)

    def draw_grid(self):
        """
        Draws the original grid and grid fills
        """
        if self.grid is not None:
            self.plot.removeItem(self.grid)

        height, width = self.parent.im.shape[:2]
        spacing_x, spacing_y = self.spacing, self.spacing

        self.grid = GridSegmentItem(spacing_x, spacing_y, width, height)
        self.plot.addItem(self.grid)

        # TODO: fix edges
        if self.grid_fills is None:
            self.grid_fills = np.empty((self.grid.num_y, self.grid.num_x), dtype=object)

        for i in range(self.grid.num_y):
            for j in range(self.grid.num_x):
                self.plot.removeItem(self.grid_fills[i, j])

                self.grid_fills[i, j] = RectangleItem((i, j), self.spacing, (height, width),
                                                      fillcolor='r')


                self.plot.addItem(self.grid_fills[i, j])

        self.grid_coord = (0, 0)
        self.plot.removeItem(self.grid_fills[0, 0])

        self.grid_fills[0, 0] = RectangleItem((0, 0), self.spacing, (height, width),
                                              fillcolor='b')

        self.plot.addItem(self.grid_fills[0, 0])

    def update_grid_fill(self, old_coord):
        """
        Updates the currently selected grid
        """
        self.plot.removeItem(self.grid_fills[old_coord])
        self.plot.removeItem(self.grid_fills[self.grid_coord])

        if self.grid_flags[old_coord]:
            new_color = 'g'
        else:
            new_color = 'r'

        height, width = self.parent.im.shape[:2]

        self.grid_fills[old_coord] = RectangleItem(old_coord, self.spacing, (height, width), fillcolor=new_color)
        self.grid_fills[self.grid_coord] = RectangleItem(self.grid_coord, self.spacing, (height, width), fillcolor='b')

        self.plot.addItem(self.grid_fills[old_coord])
        self.plot.addItem(self.grid_fills[self.grid_coord])

    def on_mouse_click(self, pos):
        """
        Updates the mouse pos in status bar

        :param pos:
        :return:
        """
        im_coords = self.viewer.mapFromScene(pos.scenePos())
        x, y = im_coords.x(), im_coords.y()

        im_x, im_y = self.parent.im.shape[:2]

        if 0 <= x <= im_x and 0 <= y <= im_y:
            old_coord = self.grid_coord
            self.grid_coord = (x // self.spacing, y // self.spacing)
            self.grid_coord = tuple(map(int, self.grid_coord))

            if self.grid_coord == old_coord:
                return

            self.parent.selector.set_im()
            self.grid_flags[self.grid_coord] = False

            self.update_grid_fill(old_coord)

    def on_button_done(self):
        """
        Updates the flag status of the overview grid
        """
        self.grid_flags[self.grid_coord] = True

        if self.grid_flags.all():
            return

        # move to next not done grid
        # but first get current position
        order = 'F'
        start = np.ravel_multi_index(self.grid_coord, self.grid_flags.shape, order=order)

        r = self.grid_flags.ravel(order=order)
        idx = np.where(r == False)[0]  # idxs of Trues

        try:
            i = np.where(idx > start)[0][0]
        except IndexError:  # unless at end
            i = 0

        idx = idx[i]  # first idx greater that start

        idx = np.unravel_index(idx, (self.grid_flags.shape), order=order)

        old_coord = self.grid_coord
        self.grid_coord = idx
        self.parent.selector.set_im()

        self.update_grid_fill(old_coord)


class SelectorViewer(GenericViewer):

    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self.sub_im = None

    def set_im(self, im=None, clear=True):
        """
        Sets image based on overviewer

        :param im:
        :param clear:
        :return:
        """
        ov = self.parent.overview
        dx, dy = ov.grid_coord
        spacing = ov.spacing

        x1, x2 = spacing*dx, spacing*(dx+1)
        y1, y2 = spacing*dy, spacing*(dy+1)

        self.sub_im = self.parent.im[y1:y2, x1:x2]
        self.viewer.setImage(self.sub_im)


# TODO: make this work
class SelectedViewer:
    """
    Grid viewer for selected axons
    """
    def __init__(self, parent):

        self.parent = parent

    def get_layout(self):
        """
        Generates the layout to pass to the scroll area
        """
        self.layout_grid = QtWidgets.QGridLayout()

        return self.layout_grid

    def update_grid(self):
        """
        Draws the pixmaps
        """
        # several test pixmaps
        test_im = self.parent.selector.sub_im[0:100, 0:100, :].copy()  # cannot be view, must be array
        height, width, channel = test_im.shape

        bpl = 3 * width  # bytes per line
        qimg = QtGui.QImage(test_im.data, width, height, bpl, QtGui.QImage.Format_RGB888)
        qpix = QtGui.QPixmap.fromImage(qimg)

        pixmaps = [QtWidgets.QLabel() for i in range(30)]

        # TODO: update layout on resize event
        for idx, p in enumerate(pixmaps):
            p.setPixmap(qpix)
            p.setFixedSize(100, 100)
            self.layout_grid.addWidget(p, idx // 2, idx % 2)  # two per row


class GridSegmentItem(pg.GraphicsObject):
    """
    Draws a pyqtgraph line segment
    """
    def __init__(self, spacing_x, spacing_y, width, height):
        pg.GraphicsObject.__init__(self)

        self.spacing_x = spacing_x
        self.spacing_y = spacing_y
        self.width = width
        self.height = height

        self.num_x = self.width // self.spacing_x + 1
        self.num_y = self.height // self.spacing_y + 1

        self.generatePicture()

    def generatePicture(self):
        self.picture = QtGui.QPicture()
        p = QtGui.QPainter(self.picture)
        p.setPen(pg.mkPen('r'))

        for i in range(self.num_x):
            x = self.spacing_x * i
            p.drawLine(QtCore.QPoint(x, 0), QtCore.QPoint(x, self.height))
        p.drawLine(QtCore.QPoint(self.width, 0), QtCore.QPoint(self.width, self.height))

        for i in range(self.num_y):
            y = self.spacing_y * i
            p.drawLine(QtCore.QPoint(0, y), QtCore.QPoint(self.width, y))
        p.drawLine(QtCore.QPoint(0, self.height), QtCore.QPoint(self.width, self.height))

        p.end()

    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return QtCore.QRectF(self.picture.boundingRect())


class RectangleItem(pg.GraphicsObject):
    """
    Draws a pyqtgraph filled rectangle
    """
    def __init__(self, top_left, size, h_w, fillcolor='r'):
        pg.GraphicsObject.__init__(self)

        self.top_left = top_left

        self.spacing_x = size
        self.spacing_y = size

        self.width = h_w[0]
        self.height = h_w[1]

        assert fillcolor in ['r', 'g', 'b']

        color = [0, 0, 0, 100]
        color[['r', 'g', 'b'].index(fillcolor)] = 255

        self.color = color

        self.generatePicture()

    def generatePicture(self):
        self.picture = QtGui.QPicture()
        p = QtGui.QPainter(self.picture)
        p.setBrush(pg.mkBrush(self.color))
        # p.setPen(pg.mkPen('w'))

        # tl = QtCore.QPointF(self.top_left[0], self.top_left[1])
        tl = QtCore.QPointF(self.top_left[0] * self.spacing_x, self.top_left[1] * self.spacing_y)

        # correct for borders
        spacing_x, spacing_y = self.spacing_x, self.spacing_y

        if self.width // self.spacing_x <= self.top_left[0]:
            spacing_x = self.width % self.spacing_x
        if self.height // self.spacing_x <= self.top_left[1]:
            spacing_y = self.height % self.spacing_y

        size = QtCore.QSizeF(spacing_x, spacing_y)

        p.drawRect(QtCore.QRectF(tl, size))
        p.end()

    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return QtCore.QRectF(self.picture.boundingRect())


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