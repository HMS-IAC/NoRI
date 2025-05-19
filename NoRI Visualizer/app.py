from __future__ import annotations
import sys
import traceback
from types import TracebackType

from qtpy.QtWidgets import QWidget, QMainWindow, QApplication, QGridLayout
from qtpy.QtGui import QIcon

from _widgets._menubar import MenuBar
from _widgets._graph_widget import GraphWidget
from _widgets._image_widget import ImageWidget
from _widgets._channel_widget import ChannelWidget

class MainWindow(QMainWindow):
    def __init__(self, parent: QWidget | None = None):

        super().__init__(parent)
        self.setWindowTitle("NoRi Visualizer")

        self.menu_bar = MenuBar(self)
        self.setMenuBar(self.menu_bar)
        
        self.graph_widget = GraphWidget(self)
        self.image_widget = ImageWidget(self)
        self.channel_widget = ChannelWidget(self)

        self.central_widget = QWidget(self)
        grid_layout = QGridLayout(self.central_widget)
        self.setCentralWidget(self.central_widget)
        
        grid_layout.addWidget(self.image_widget, 0, 0)
        grid_layout.addWidget(self.graph_widget, 0, 1)
        grid_layout.addWidget(self.channel_widget, 1, 0, 1, 2)

        # connections
        self.image_widget.comboChanged.connect(self._update_widget_info)
        self.image_widget.roiChanged.connect(self._highlight_point_in_graph)
        self.graph_widget.pointSelected.connect(self._highlight_roi_in_vispy_canvas)

    def _update_widget_info(self, image_name: str) -> None:
        self.image_widget.viewer._clear_highlight()
        self.graph_widget.set_dataframe(image_name)
        self.channel_widget.loadRawImage(image_name)
        self.image_widget.viewer._reset()
        self.channel_widget.figure.clear()
        self.channel_widget.canvas.draw()
    
    def _highlight_point_in_graph(self, roi: int) -> None:
        if not roi:
            self.channel_widget.figure.clear()
            self.channel_widget.canvas.draw()
            return
        point = self._roi_to_point(roi)
        if point is None:
            self.channel_widget.figure.clear()
            self.channel_widget.canvas.draw()
            return
        self.graph_widget.highlight_point(roi)
        self.channel_widget.update_channel(roi)
        
    def _highlight_roi_in_vispy_canvas(self, info: tuple[float, float, int]) -> None:
        _, _, roi = info
        self.image_widget.viewer._highlight_rois(roi)
        self.channel_widget.update_channel(roi)

    def _roi_to_point(self, roi: int) -> tuple[float, float] | None:
        if not roi:
            return None
        try:
            data = self.graph_widget.data_df[self.graph_widget.data_df.id==roi]
            plot_type = self.graph_widget.combobox.currentText()
            x = data.mean_protein.values[0] if plot_type == "Scatter plot" else 1
            y = data.mean_lipid.values[0]
        except IndexError:
            return None
        
        return x, y


def _our_excepthook(
    type: type[BaseException], value: BaseException, tb: TracebackType | None
) -> None:
    """Excepthook that prints the traceback to the console.

    By default, Qt's excepthook raises sys.exit(), which is not what we want.
    """
    # this could be elaborated to do all kinds of things...
    traceback.print_exception(type, value, tb)


if __name__ == "__main__":
    app = QApplication([])
    app.setWindowIcon(QIcon('assets/icon/nori.png'))
    mw = MainWindow()
    mw.show()
    # this is to avoid the application crashing. if an error occurs, it will be printed
    # to the console instead.
    sys.excepthook = _our_excepthook
    app.exec()