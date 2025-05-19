from __future__ import annotations
import tifffile
import cv2 as cv
from skimage.measure import label
from qtpy.QtWidgets import (
    QGroupBox,
    QLabel,
    QWidget,
    QHBoxLayout,
    QFormLayout,
)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
import pandas as pd
from _widgets.utils import extract_tubule
from _widgets.utils import DATA_PATH

ROI = "Tubule number: "
MPC = "Mean protein concentration: "
MLC = "Mean lipid concentration: "
NN = "Number of nuclei: "
BBE = "Brush border exists: "
TT = "Tubule type: "


class ChannelWidget(QGroupBox):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)

        stats = QGroupBox("Tubule statistics")
        self.label = QLabel()
        self.mean_protein = QLabel()
        self.mean_lipid = QLabel()
        self.nuclei = QLabel()
        self.brush = QLabel()
        self.tubule_type = QLabel()
        stats_layout = QFormLayout(stats)
        stats_layout.setContentsMargins(10, 10, 10, 10)
        stats_layout.setSpacing(5)
        stats_layout.addRow(ROI, self.label)
        stats_layout.addRow(MPC, self.mean_protein)
        stats_layout.addRow(MLC, self.mean_lipid)
        stats_layout.addRow(NN, self.nuclei)
        stats_layout.addRow(BBE, self.brush)
        stats_layout.addRow(TT, self.tubule_type)

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        main_layout.addWidget(self.canvas, 1)
        main_layout.addWidget(stats, 0)

    def loadRawImage(self, image_name: str) -> None:
        self.figure.clear()
        raw_image_path = f"{DATA_PATH['raw_images']}/{image_name}.tif"
        self.raw_image = tifffile.imread(raw_image_path)
        self.num_channels, _, _ = self.raw_image.shape

        tubule_image_path = f"{DATA_PATH['tubule_labels']}/{image_name}.png"
        self.tubules = cv.imread(tubule_image_path, cv.IMREAD_UNCHANGED)

        full_image_path = f"{DATA_PATH['processed_images']}/{image_name}.tif"
        self.image = tifffile.imread(full_image_path)

        self.data_df = pd.read_csv(f"{DATA_PATH['csv_files']}/{image_name}.csv")
    
    def update_channel(self, roi: int) -> None:
        self._display_images(roi)
        self._set_stats(roi)

    def _set_stats(self, roi) -> None:
        data = self.data_df[self.data_df.id==roi]
        self.mean_protein.setText(f"{data.mean_protein.values[0]:.2f}")
        self.mean_lipid.setText(f"{data.mean_lipid.values[0]:.2f}")
        self.nuclei.setText(f"{data.nuclei_count.values[0]}")
        self.brush.setText("Yes" if data.bb_exists.values[0] else "No")
        self.tubule_type.setText(data.tubule_class.values[0])

    def _display_images(self, roi: int):
        self.label.setText(f"{roi}")
        self.figure.clear()

        ch_mask, x, y, w, h = extract_tubule(self.tubules, roi)

        # Display 6 channels
        for i in range(0, self.num_channels):
            ch = cv.bitwise_and(
                self.raw_image[i, y : y + h, x : x + w],
                self.raw_image[i, y : y + h, x : x + w],
                mask=ch_mask.astype(np.uint8),
            )
            ax = self.figure.add_subplot(1, self.num_channels + 1, i + 1)
            # flip along y-axis to match the orientation of the original image
            ch = cv.flip(ch, 0)
            ax.imshow(ch, cmap="gray")
            if i==0:
                ax.set_title(f"Protein channel")
            elif i==1:
                ax.set_title(f"Lipid channel")
            else:
                ax.set_title(f"Ch {i+1}")
            
            ax.axis("off")

        ch = cv.bitwise_and(
            self.image[y : y + h, x : x + w],
            self.image[y : y + h, x : x + w],
            mask=ch_mask.astype(np.uint8),
        )
        # flip along y-axis to match the orientation of the original image
        ch = cv.flip(ch, 0)
        ax = self.figure.add_subplot(1, self.num_channels + 1, self.num_channels + 1)
        ax.imshow(ch)
        ax.set_title("Protein and Lipid")
        ax.axis("off")

        self.figure.tight_layout(pad=0.3)

        self.canvas.draw()
