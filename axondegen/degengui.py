"""
Test of gui
"""

import logging
import sys
import json
import traceback as tb
from pathlib import Path
from itertools import chain

from PIL import Image
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore, QtGui
# from flowlayout import FlowLayout

# from imageprocess import ImageProcess

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

        self.setGeometry(250, 100, 1000, 750)
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
        self.im_path = None
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

        # selector tab
        layout_selector = self.setup_selector()
        self.sel_tab = QtWidgets.QWidget()
        self.tabs.addTab(self.sel_tab, 'Select')
        self.sel_tab.setLayout(layout_selector)

        # viewer tab
        layout_viewer = self.setup_viewer()
        self.viewer_tab = QtWidgets.QWidget()
        self.tabs.addTab(self.viewer_tab, 'Viewer')
        self.viewer_tab.setLayout(layout_viewer)

        self.image_viewer.set_im(None)

        # top level layout
        layout_frame = QtWidgets.QHBoxLayout()

        layout_frame.addWidget(self.tabs)
        # layout_frame.addLayout(layout_controls)

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

        :return: layout with viewer
        """
        self.image_viewer = AllViewer(self)
        # hackish bad fix
        self.image_viewer.sizeHint = lambda: QtCore.QSize(600, 600)
        self.image_viewer.scene().sigMouseMoved.connect(self.image_viewer.on_mouse_move)
        # self.image_viewer.scene().sigMouseClicked.connect(self.on_mouse_click)

        # viewer layout
        layout_viewer = QtWidgets.QHBoxLayout()
        layout_viewer.addWidget(self.image_viewer)

        return layout_viewer

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

        # layout for buttons
        layout_buttons = QtWidgets.QHBoxLayout()

        self.button_done = QtWidgets.QPushButton('Done')
        self.button_done.clicked.connect(self.overview.on_button_done)

        self.button_save = QtWidgets.QPushButton('Save')
        self.button_save.clicked.connect(self.overview.on_button_save)

        self.button_load = QtWidgets.QPushButton('Load')
        self.button_load.clicked.connect(self.overview.on_button_load)

        self.degen_count = QtWidgets.QLabel('Count: {} '.format(0))

        layout_buttons.addWidget(self.button_done)
        layout_buttons.addWidget(self.button_save)
        layout_buttons.addWidget(self.button_load)

        layout_left_controls.addLayout(layout_buttons)
        layout_left_controls.addWidget(self.degen_count)
        layout_left_controls.setAlignment(QtCore.Qt.AlignLeft)

        layout_left_panel.addLayout(layout_left_controls)

        # right side cell selector
        self.selector = SelectorViewer(self)

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

        # self.selector.sigMouseDragged.connect(self.selector.on_mouse_drag)
        # self.selector.sigMouseDragged.scene().connect(self.selector.on_mouse_drag)
        # self.selector.mouseDragEvent.connect(self.selector.on_mouse_drag)

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

        self.im_path = im_path

        im = Image.open(str(im_path))

        # if 2D array (grayscale, not RGB), make 3D RGB
        if len(im.shape) == 2:
            im = np.stack((im,)*3, -1)

        self.im = im


        self.image_viewer.set_im(self.im)

    def on_button_find_degen(self):
        """
        Handles getting degenerated centers

        :return:
        """
        pass
        # self.im_process = ImageProcess(self.im)
        #
        # # get mask
        # m = self.im_process.get_degen_mask('dt', thresh=self.thresh_value)
        # c = self.im_process.get_centroids(m, min_dist=30)
        #
        # self.centers = c
        #
        # self.status_count.setText('Count: {} '.format(len(c)))
        #
        # if self.current_view == 'binary':
        #     self.image_viewer.set_im(self.im_process.thresh)
        # elif self.current_view == 'method':
        #     self.image_viewer.set_im(self.im_process.method)
        # elif self.current_view == 'result':
        #     self.image_viewer.set_im(self.im_process.result)
        #
        # self.image_viewer.plot_centers(self.centers)

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

        :param im:
        :param clear:
        :return:
        """
        self.viewer.setImage(im)

        # TODO: figure out aesthetics here to get rid of border
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

        self.rects = []

    def set_im(self, im=None, clear=True):
        """
        Updates the viewer to an image
        """
        if im is None:
            try:
                im_path = r'..\docs\sample_data\slide01_section1_area08.tif'
                im_path = Path(im_path)
                assert im_path.exists(), f'cannot find image {im_path}'
            except AssertionError:
                return

            self.parent.im_path = im_path
            im = Image.open(str(im_path))
            im = np.stack((im,) * 3, -1)
            self.parent.im = im

        if clear:
            self.scatter.clear()
            self.parent.status_count.setText('Count: {} '.format(0))

            # self.parent.checkbox_method.blockSignals(True)
            # self.parent.checkbox_method.setChecked(False)
            # self.parent.checkbox_method.blockSignals(False)

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

    def clear_rects(self):
        """
        Clears out the currently drawn selections
        """
        vb = self.viewer.getViewBox()
        for rect in self.rects:
            vb.removeItem(rect)

        self.rects = []

    def draw_rect(self, rect):
        """
        Draws the permanent rect that stays after the end of the drag event
        """
        # just reuse viewbox items instead of recreating our own
        vb = self.viewer.getViewBox()

        grid_rect = QtGui.QGraphicsRectItem(*rect)
        grid_rect.setPen(pg.mkPen((255,0,100), width=1))
        grid_rect.setBrush(pg.mkBrush(255,0,0,50))
        grid_rect.setZValue(1e9)
        grid_rect.show()
        vb.addItem(grid_rect, ignoreBounds=True)

        self.rects.append(grid_rect)


class AllViewer(GenericViewer):

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.parent.selector.rects_changed.connect(self.update_selections)

    def update_selections(self):
        """
        draws grids
        """
        to_draw = self.parent.overview.grid_rects_to_ar()

        self.clear_rects()
        for rect in to_draw:
            self.draw_rect(rect)


class OverViewer(GenericViewer):

    # signals
    sigUpdateRects = QtCore.Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self.grid_coord = (0, 0)
        # self.spacing = 550
        self.spacing = 784

        self.grid = None
        self.grid_rects = None
        self.grid_fills = None

        self.viewer.getViewBox().keyPressEvent = self.on_key_press

    def set_im(self, im=None, clear=True):
        """
        Updates the viewer to an image

        :return:
        """
        self.viewer.setImage(self.parent.im)
        self.draw_grid()

        self.grid_flags = np.full((self.grid.num_y, self.grid.num_x), fill_value=False, dtype=bool)
        self.grid_rects = np.empty((self.grid.num_y, self.grid.num_x), dtype=np.ndarray)
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

    def on_key_press(self, ev):
        """
        Hijacks key press event
        """
        if ev.key() == QtCore.Qt.Key_Space:
            self.on_button_done()

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
            self.on_button_save()
            return

        # move to next not done grid
        # but first get current position
        start = np.ravel_multi_index(self.grid_coord, self.grid_flags.shape, order='F')

        r = self.grid_flags.ravel(order='F')
        idx = np.where(r == False)[0]  # idxs of Trues

        try:
            i = np.where(idx > start)[0][0]
        except IndexError:  # unless at end
            i = 0

        idx = idx[i]  # first idx greater that start

        idx = np.unravel_index(idx, (self.grid_flags.shape), order='F')

        old_coord = self.grid_coord
        self.grid_coord = idx
        self.parent.selector.set_im()

        self.update_grid_fill(old_coord)

    def grid_rects_to_ar(self):
        """
        Converts from the grid rect format to a flattened array of image coords
        """
        out_grid = self.grid_rects.copy()

        def add_to_tuple(t, dx, dy):
            t = tuple(map(round, t))
            a, b, c, d = t
            return (a+dx, b+dy, c, d)

        for idx, i in enumerate(self.grid_rects):
            dx = self.spacing * idx

            for idx2, j in enumerate(i):
                dy = self.spacing * idx2

                if j is not None:
                    out_grid[idx, idx2] = [add_to_tuple(k, dx, dy) for k in j]

        out_grid = [i for i in out_grid.flatten() if i is not None]
        out_grid = list(chain.from_iterable(out_grid))

        return out_grid

    def on_button_save(self):
        """
        Saves a text file with grid coord info for each selected degen axon
        """
        if self.grid_rects.any():
            save_name = self.parent.im_path.stem
            save_dir = self.parent.im_path.parents[0] / save_name

            # get save path
            save_path = QtWidgets.QFileDialog.getSaveFileName(self, 'Save coordinates', str(save_dir),
                                                                     filter='JSON (*.json);;'\
                                                                            'All files (*)')[0]
            if not save_path:
                return

            out = self.grid_rects_to_ar()

            with open(save_path, 'w') as f:
                json.dump(out, f, indent=2)

    def on_button_load(self):
        """
        Saves a text file with grid coord info for each selected degen axon
        """
        # raise NotImplementedError('loading not yet set up')

        load_path = QtWidgets.QFileDialog.getOpenFileName(self, 'Open image',
                                                                filter='JSON (*.json);;'\
                                                                'All files (*)')[0]
        if not load_path:
            return

        load_path = Path(load_path)
        assert load_path.exists()

        with open(load_path, 'r') as f:
            raw_data = json.load(f)

        def get_mods(t, spacing):
            x, y, dx, dy = t
            return (x % spacing, y % spacing, dx, dy)

        def get_idxs(t, spacing):
            x, y, dx, dy = t
            return (x // spacing, y // spacing)

        data = [get_mods(i, self.spacing) for i in raw_data]
        idxs = [get_idxs(i, self.spacing) for i in raw_data]

        self.parent.selector.clear_rects()

        self.grid_flags = np.full((self.grid.num_y, self.grid.num_x), fill_value=False, dtype=bool)
        self.draw_grid()
        self.grid_rects = np.empty((self.grid.num_y, self.grid.num_x), dtype=np.ndarray)

        for idx, rect in zip(idxs, data):
            self.parent.selector.add_rect_to_rects(rect, idx)

        self.parent.selector.draw_existing_rects()


class SelectorViewer(GenericViewer):

    rects_changed = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self.sub_im = None

        self.selected_rect = None

        self.viewer.getViewBox().mouseClickEvent = self.on_mouse_click
        self.viewer.getViewBox().mouseDragEvent = self.on_mouse_drag
        self.viewer.getViewBox().keyPressEvent = self.on_key_press

        self.rects_changed.connect(self.update_rects)

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

        self.clear_rects()
        self.draw_existing_rects()

    def clear_rects(self):
        """
        Clears out the currently drawn selections
        """
        super().clear_rects()

        self.selected_rect = None

        self.rects_changed.emit()

    def update_rects(self):
        """
        Updates rect count
        """
        self.parent.degen_count.setText('Count: {} '.format(len(self.rects)))

        total_count = 0
        for a in self.parent.overview.grid_rects.flatten():
            if a is not None:
                total_count += len(a)

        self.parent.status_count.setText('Count: {} '.format(total_count))

    def draw_existing_rects(self):
        """
        Draws the rects for that grid coord
        """
        ov = self.parent.overview
        grid_rects = ov.grid_rects[ov.grid_coord]

        if grid_rects is not None:
            for rect in grid_rects:

                grid_rect = self.draw_rect(rect)
                self.rects.append(grid_rect)

        self.rects_changed.emit()

    def add_rect_to_rects(self, rect, idx=None):
        """
        Adds a rect to rect container
        :param rect:
        :return:
        """
        ov = self.parent.overview
        grid_rects = ov.grid_rects

        if idx is None:
            idx = ov.grid_coord

        if grid_rects[idx] is None:
            grid_rects[idx] = [rect]
        else:
            grid_rects[idx].append(rect)

    def draw_new_rect(self, rect):
        """
        Handles drawing of new rects
        """
        if isinstance(rect, QtCore.QRectF):
            rect = list(rect.getRect())

        # don't draw rects that are completely out of bounds
        if not rect[0] + rect[2] > 0 or not rect[1] + rect[3] > 0:
            return

        shape = self.sub_im.shape
        if rect[0] > shape[1] or rect[1] > shape[0]:
            return

        grid_rect = self.draw_rect(rect)

        self.rects.append(grid_rect)

        ov = self.parent.overview
        grid_rects = ov.grid_rects

        self.add_rect_to_rects(grid_rect.rect().getRect())

        self.rects_changed.emit()

    def draw_rect(self, rect):
        """
        Draws the permanent rect that stays after the end of the drag event
        """
        # just reuse viewbox items instead of recreating our own
        vb = self.viewer.getViewBox()
        ov = self.parent.overview

        if isinstance(rect, QtCore.QRectF):
            fixed_size = list(rect.getRect())
        else:
            fixed_size = rect

        # stop the selection going out of bounds on top and left
        if fixed_size[0] < 0:
            fixed_size[2] += fixed_size[0]
        if fixed_size[1] < 0:
            fixed_size[3] += fixed_size[1]
        fixed_size = np.clip(fixed_size, 0, None)

        spacing = [self.parent.overview.spacing] * 2

        # stop the selection going out of bounds on bottom and right
        # determine if on edge piece (right and bottom)
        edge = np.array((ov.grid.num_y, ov.grid.num_x)) - 1 - np.array(ov.grid_coord)
        if not np.all(edge):
            for idx, e in enumerate(edge):
                if e == 0:
                    # calc size of edge piece, make sure not zero
                    new_spacing = self.parent.im.shape[idx] % spacing[idx]
                    if new_spacing != 0:
                        spacing[idx] = new_spacing

        # don't let box go out of bounds
        for i in range(2):
            if fixed_size[0+i] + fixed_size[2+i] > spacing[i]:
                fixed_size[2+i] = spacing[i] - fixed_size[0+i]

        grid_rect = QtGui.QGraphicsRectItem(*fixed_size)
        grid_rect.setPen(pg.mkPen((255,0,100), width=1))
        grid_rect.setBrush(pg.mkBrush(255,0,0,50))
        grid_rect.setZValue(1e9)
        grid_rect.show()
        vb.addItem(grid_rect, ignoreBounds=True)

        if self.selected_rect is not None:
            self.color_rects(self.selected_rect, 'red')
            self.selected_rect = None

        return grid_rect

    def color_rects(self, rects, color):
        """
        Recolors rects
        """
        assert color in ['red', 'blue']
        if color == 'red':
            pen = pg.mkPen((255, 0, 100), width=1)
            brush = pg.mkBrush(255, 0, 0, 50)
        elif color == 'blue':
            pen = pg.mkPen((100, 0, 255), width=1)
            brush = pg.mkBrush(0, 0, 255, 50)

        if isinstance(rects, QtWidgets.QGraphicsRectItem):
            rects = [rects]

        for rect in rects:
            rect.setPen(pen)
            rect.setBrush(brush)
            rect.update()

    def on_mouse_click(self, ev):
        """
        Hijacks middle button click to autoRange, and left button to select rect
        """
        if ev.button() & QtCore.Qt.MidButton:
            ev.accept()
            self.viewer.getViewBox().autoRange()

        elif ev.button() & QtCore.Qt.LeftButton:
            ev.accept()
            clicked = self.viewer.scene().items(ev.scenePos())
            rects = [i for i in clicked if isinstance(i, QtWidgets.QGraphicsRectItem)]

            if rects:
                areas = [rect.rect().getRect() for rect in rects]
                areas = [area[2]*area[3] for area in areas]
                idx_min = areas.index(min(areas))

                # uncolor if one is already selected
                if self.selected_rect is not None:
                    self.color_rects(self.selected_rect, 'red')

                rect = rects[idx_min]
                self.selected_rect = rect
                self.color_rects(self.selected_rect, 'blue')

            else:
                self.color_rects(self.rects, 'red')
                self.selected_rect = None

    def on_key_press(self, ev):
        """
        Hijacks key press event
        """
        if ev.key() == QtCore.Qt.Key_Delete or ev.key() == QtCore.Qt.Key_D:
            if self.selected_rect is not None:
                idx = self.rects.index(self.selected_rect)

                self.viewer.getViewBox().removeItem(self.selected_rect)
                del(self.rects[idx])

                ov = self.parent.overview
                del(ov.grid_rects[ov.grid_coord][idx])
                if not ov.grid_rects[ov.grid_coord]:
                    ov.grid_rects[ov.grid_coord] = None

                self.rects_changed.emit()

                self.selected_rect = None

        if ev.key() == QtCore.Qt.Key_Space:
            self.parent.overview.on_button_done()

    def on_mouse_drag(self, ev, axis=None):
        """
        Hijacks viewbox mouse drag event to make on own drag box
        """
        ev.accept()
        pos = ev.pos()
        lastPos = ev.lastPos()
        dif = pos - lastPos
        dif = dif * -1

        # just reuse viewbox items instead of recreating our own
        vb = self.viewer.getViewBox()

        # scale or translate based on mouse button
        if ev.button() & QtCore.Qt.LeftButton:

            if ev.isFinish():  # this is the final move in the drag; change the view scale now
                vb.rbScaleBox.hide()
                ax = QtCore.QRectF(pg.Point(ev.buttonDownPos(ev.button())), pg.Point(pos))
                ax = vb.childGroup.mapRectFromParent(ax)

                self.draw_new_rect(ax)

            else:
                # update shape of scale box
                vb.updateScaleBox(ev.buttonDownPos(), ev.pos())

        elif ev.button() & QtCore.Qt.RightButton:
            tr = dif
            tr = vb.mapToView(tr) - vb.mapToView(pg.Point(0, 0))
            x = tr.x()
            y = tr.y()

            vb._resetTarget()
            if x is not None or y is not None:
                vb.translateBy(x=x, y=y)
                vb.sigRangeChangedManually.emit(vb.state['mouseEnabled'])


class SelectedViewer:
    """
    Grid viewer for selected axons
    """
    def __init__(self, parent):

        self.parent = parent
        self.parent.selector.rects_changed.connect(self.update_grid)

    def get_layout(self):
        """
        Generates the layout to pass to the scroll area
        """
        self.layout_grid = QtWidgets.QGridLayout()

        # TODO: figure out flow layout
        # self.layout_grid = FlowLayout()
        self.layout_grid.setSizeConstraint(3)

        return self.layout_grid

    def update_grid(self):
        """
        Draws the pixmaps
        """
        # first clear out existing pixmaps
        while self.layout_grid.count():
            item = self.layout_grid.takeAt(0)
            widget = item.widget()
            widget.deleteLater()

        # get bounding rects
        sub_im = self.parent.selector.sub_im
        rects = self.parent.selector.rects

        # get coords from objects
        rects = [rect.rect().getRect() for rect in rects]
        rects = [list(map(int, rect)) for rect in rects]

        # second half is distance
        coord = lambda c: [c[0], c[1], c[0]+c[2], c[1]+c[3]]
        rects = [coord(rect) for rect in rects]

        ims = [sub_im[rect[1]:rect[3], rect[0]:rect[2]].copy() for rect in rects]

        cols = 2
        rows = (len(ims)+1) // cols

        for idx, im in enumerate(ims):
            height, width, channel = im.shape
            bpl = 3 * width

            qimg = QtGui.QImage(im.data, width, height, bpl, QtGui.QImage.Format_RGB888)
            qpix = QtGui.QPixmap.fromImage(qimg)

            if height > 100 or width > 100:
                qpix = qpix.scaled(64, 64, QtCore.Qt.KeepAspectRatio)

            pixmap = QtWidgets.QLabel()

            pixmap.setPixmap(qpix)

            row = idx // cols
            col = idx % cols

            self.layout_grid.addWidget(pixmap, row, col)  # two per row
            # self.layout_grid.addWidget(pixmap)

        # print(self.parent.overview.grid_rects, end='\n\n')

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

        self.num_x = self.width // self.spacing_x
        self.num_y = self.height // self.spacing_y

        if self.width % self.spacing_x != 0:
            self.num_x += 1
        if self.height % self.spacing_y != 0:
            self.num_y += 1

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