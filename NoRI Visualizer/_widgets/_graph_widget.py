from __future__ import annotations
import contextlib
from typing import Callable

import numpy as np
from qtpy.QtWidgets import (
    QGroupBox,
    QComboBox,
    QLabel,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import pandas as pd
import mplcursors
from matplotlib.axes import Axes
from qtpy.QtCore import Signal

from _widgets.utils import DATA_PATH

SCATTER_PLOT = "Scatter plot"
BOX_PLOT = "Box plot"



# COLOR_MAPPING = {
#     "LTL": (1,0,0,0.5),
#     "Umod": (0,1,0,0.5),
#     "AQP2": (0,0,1,0.5),
#     "Unlabeled": (0,1,1,0.5)
# }

COLOR_MAPPING = {
    "LTL+": (0.0, 0.50, 0.0, 0.5),
    "Umod+": (1.0, 0.65, 0.0, 0.5),
    "AQP2+": (0.94, 0.44, 0.99, 0.5),
    "Unlabeled": (0.50, 0.50, 0.50, 0.5)
}


class GraphWidget(QGroupBox):

    pointSelected = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.data_df: pd.DataFrame | None = None

        self.on_add: Callable | None = None

        self.roi: int | None = None

        self.scatter: Axes | None = None

        self.box: Axes | None = None

        self.combobox = QComboBox()
        self.combobox.addItems(["", SCATTER_PLOT, BOX_PLOT])
        self.combobox.currentTextChanged.connect(self._plot)

        label = QLabel("Graph type:")
        top_layout = QHBoxLayout()
        top_layout.setContentsMargins(10, 10, 10, 10)
        top_layout.setSpacing(5)
        top_layout.addWidget(label, stretch=0)
        top_layout.addWidget(self.combobox, stretch=1)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(5)

        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.canvas)

    def set_dataframe(self, image_name: str) -> None:
        self.data_df = pd.read_csv(f"{DATA_PATH['csv_files']}/{image_name}.csv")

        if text := self.combobox.currentText():
            self._plot(text)

    def _plot(self, plot_type: str) -> None:
        self.figure.clear()

        if plot_type == "" or self.data_df is None:
            self.canvas.draw()
            return

        if plot_type == SCATTER_PLOT:
            data = self.data_df[["id", "mean_protein", "mean_lipid", "tubule_class"]]
            # data = data[(data.mean_protein != 0) & (data.mean_lipid != 0)]
            ax = self.figure.add_subplot(1, 1, 1)
            colors = [COLOR_MAPPING[cls] for cls in data["tubule_class"]]
            self.scatter = ax.scatter(data.mean_protein, data.mean_lipid, c=colors)
            ax.set_xlabel("Mean Protein (mg/ml)")
            ax.set_ylabel("Mean Lipid (mg/ml)")
            ax.set_title("Mean protein vs lipid concentration per tubule")
            temp_data = data[(data.mean_protein != 0) & (data.mean_lipid != 0)]
            xlow = temp_data.mean_protein.min() - 5
            xhigh = temp_data.mean_protein.max() + 5
            ylow = temp_data.mean_lipid.min() - 5
            yhigh = temp_data.mean_lipid.max() + 5

            ax.set_xlim(xlow, xhigh)
            ax.set_ylim(ylow, yhigh)

            for label, color in COLOR_MAPPING.items():
                ax.scatter([], [], color=color, label=label, alpha=0.5)
            # ax.legend(loc='lower center', bbox_to_anchor=(0.5, 0.1), ncol=4)
            ax.legend(loc='lower center', ncol=4)

        elif plot_type == BOX_PLOT:
            data = self.data_df[["id", "std_lipid", "tubule_class"]]
            data = data[(data.std_lipid != 0)]
            ax = self.figure.add_subplot(1, 1, 1)
            ax.boxplot(data.std_lipid, showfliers=False)
            colors = [COLOR_MAPPING[cls] for cls in data["tubule_class"]]
            jittered_x = np.random.normal(1, 0.04, size=len(data))
            self.box = ax.scatter(jittered_x, data.std_lipid, c=colors)
            ax.set_ylabel("STD Lipid (mg/ml)")
            ax.set_title("STD lipid concentration per tubule")

            for label, color in COLOR_MAPPING.items():
                ax.scatter([], [], color=color, label=label, alpha=0.5)
            ax.legend(loc='upper right')

        cursor = mplcursors.cursor(ax)

        @cursor.connect("add")  # type: ignore [misc]
        def on_add(sel: mplcursors.Selection, signal: bool = True) -> None:
            with contextlib.suppress(AttributeError):
                sel.annotation.set_visible(False)
            if plot_type:
                graph = self.scatter if plot_type == SCATTER_PLOT else self.box
                # reset all face colors to green and set the selected point to magenta
                colors = [COLOR_MAPPING[cls] for cls in data["tubule_class"]]
                # colors = ["green"] * len(self.scatter.get_offsets())
                colors[sel.index] = (0,0,0,1)
                graph.set_color(colors)
                self.canvas.draw_idle()
                roi = sel.index+1

            if signal:
                info = (sel.target[0], sel.target[1], roi)
                self.pointSelected.emit(info)

        self.on_add = on_add

        self.canvas.draw()

    def highlight_point(self, roi: int) -> None:
        plot_type = self.combobox.currentText()
        
        if plot_type == "":
            return
        
        graph = self.scatter if plot_type == SCATTER_PLOT else self.box
        
        if graph is None:
            return
        
        class MockSelection:
            def __init__(self, artist, index):
                self.artist = artist
                self.index = index
                self.target = artist.get_offsets()[index]
        
        # Create a mock selection and call on_add
        mock_selection = MockSelection(graph, roi-1)
        self.on_add(mock_selection, signal=False)
