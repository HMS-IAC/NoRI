from __future__ import annotations

from typing import Any
import tifffile
from qtpy.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QComboBox,
)
from qtpy.QtCore import Signal
from _widgets.utils import _create_mask_outlines, read_file_names
from _widgets._viewer import Viewer
import cv2 as cv
from _widgets.utils import DATA_PATH
from qtpy.QtCore import Qt

from ndv import NDViewer, DataWrapper


class _NDViewer(NDViewer):
    """NDViewer subclass to hide 3D functionality and emit a signal when closed."""

    closed = Signal(str)

    def __init__(self, data: DataWrapper | Any, title: str, *args: Any, **Kwargs: Any):
        super().__init__(data, *args, **Kwargs)
        self.setWindowTitle(title)
        # set as a dialog window
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.Dialog)
        # hide 3D button
        self._ndims_btn.hide()

        self._title = title

    def closeEvent(self, event: Any) -> None:
        self.closed.emit(self._title)
        super().closeEvent(event)
        

class ImageWidget(QGroupBox):

    comboChanged = Signal(object)
    roiChanged = Signal(object)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        self._raw_images: dict[str, _NDViewer] = {}

        # file combobox
        self.files_combo = QComboBox()
        self.file_names = read_file_names(DATA_PATH["raw_images"])
        self.file_names.insert(0, "")
        self.files_combo.addItems(self.file_names)

        # show raw data button
        self._show_raw_data_btn = QPushButton("Show raw image")

        top_group = QGroupBox()
        top_layout = QHBoxLayout(top_group)
        top_layout.setContentsMargins(10, 10, 10, 10)
        top_layout.setSpacing(10)
        top_layout.addWidget(QLabel("File Name:"), 0)
        top_layout.addWidget(self.files_combo, 1)
        top_layout.addWidget(self._show_raw_data_btn, 0)

        # viewer
        self.viewer = Viewer(self)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(5)
        main_layout.addWidget(top_group)
        main_layout.addWidget(self.viewer)

        # connections
        self._show_raw_data_btn.clicked.connect(self._show_raw_data)
        self.files_combo.currentTextChanged.connect(self._init_widget)
        self.viewer.roiChanged.connect(self.roiChanged)
        
    def _init_widget(self, image_name: str):

        full_image_path = f"{DATA_PATH['processed_images']}/{image_name}.tif"
        image = tifffile.imread(full_image_path)

        tubule_image_path = f"{DATA_PATH['tubule_labels']}/{image_name}.png"
        tubules = cv.imread(tubule_image_path, cv.IMREAD_UNCHANGED)

        nuclei_image_path = f"{DATA_PATH['nuclei_masks']}/{image_name}.png"
        nuclei = cv.imread(nuclei_image_path, cv.IMREAD_GRAYSCALE)
        nuclei = _create_mask_outlines(nuclei)

        lumen_image_path = f"{DATA_PATH['lumen_masks']}/{image_name}.png"
        lumen = cv.imread(lumen_image_path, cv.IMREAD_GRAYSCALE)
        lumen = _create_mask_outlines(lumen)

        bb_image_path = f"{DATA_PATH['bb_masks']}/{image_name}.png"
        bb = cv.imread(bb_image_path, cv.IMREAD_GRAYSCALE)
        bb = _create_mask_outlines(bb)

        glomerulus_image_path = f"{DATA_PATH['glomeruli_masks']}/{image_name}.png"
        glomerulus = cv.imread(glomerulus_image_path, cv.IMREAD_GRAYSCALE)
        glomerulus = _create_mask_outlines(glomerulus)

        self.viewer.setData(image, tubules, nuclei, bb, lumen, glomerulus)

        self.comboChanged.emit(image_name)

    def _show_raw_data(self) -> None:
        image_name = self.files_combo.currentText()
        if not image_name:
            return
        
        # show the image if it's already in the dict
        if image_name in self._raw_images:
            ndviewer = self._raw_images[image_name]
            ndviewer.raise_() if ndviewer.isVisible() else ndviewer.show()
            return

        # otherwise, load the image
        full_image_path = f"{DATA_PATH['raw_images']}/{image_name}.tif"
        image = tifffile.imread(full_image_path)
        image = image[:, ::-1, :]
        ndviewer = _NDViewer(image, title=image_name, parent=self)
        ndviewer.closed.connect(self._on_ndviewer_closed)
        self._raw_images[image_name] = ndviewer
        ndviewer.show()

    def _on_ndviewer_closed(self, image_name: str) -> None:
        self._raw_images.pop(image_name)
