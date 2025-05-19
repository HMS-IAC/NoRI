from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from fonticon_mdi6 import MDI6
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from vispy.color import Colormap
from superqt import QLabeledRangeSlider
from superqt.fonticon import icon
from superqt.utils import qthrottled, signals_blocked
from vispy import scene


if TYPE_CHECKING:
    from typing import Literal

    from vispy.scene.events import SceneMouseEvent

IMAGE = "Image"
TUBULES = "Tubules"
NUCLEI = "Nuclei"
BORDERS = "Brush Borders"
LUMEN = "Lumen"
GLOMERULUS = "Glomerulus"


SS = """
QSlider::groove:horizontal {
    height: 15px;
    background: qlineargradient(
        x1:0, y1:0, x2:0, y2:1,
        stop:0 rgba(128, 128, 128, 0.25),
        stop:1 rgba(128, 128, 128, 0.1)
    );
    border-radius: 3px;
}

QSlider::handle:horizontal {
    width: 38px;
    background: #999999;
    border-radius: 3px;
}

QLabel { font-size: 12px; }

QRangeSlider { qproperty-barColor: qlineargradient(
        x1:0, y1:0, x2:0, y2:1,
        stop:0 rgba(100, 80, 120, 0.2),
        stop:1 rgba(100, 80, 120, 0.4)
    )}

SliderLabel {
    font-size: 12px;
    color: white;
}
"""


def show_error_dialog(parent: QWidget, message: str) -> None:
    """Show an error dialog with the given message."""
    dialog = QMessageBox(parent)
    dialog.setWindowTitle("Error")
    dialog.setText(message)
    dialog.setIcon(QMessageBox.Icon.Critical)
    dialog.setStandardButtons(QMessageBox.StandardButton.Ok)
    dialog.exec()


class Viewer(QGroupBox):
    """A widget for displaying an image."""

    roiChanged = Signal(object)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent=parent)

        self._viewer = _ImageCanvas(parent=self)

        # ROIs -------------------------------------------------------------

        # roi number indicator
        find_roi_lbl = QLabel("Tubules ROI:")
        self._roi_number = QLabel()
        self._clear_btn = QPushButton("Clear Selected")
        self._clear_btn.clicked.connect(self._clear_highlight)
        roi_wdg = QWidget()
        roi_layout = QHBoxLayout(roi_wdg)
        roi_layout.setContentsMargins(0, 0, 0, 0)
        roi_layout.addWidget(find_roi_lbl)
        roi_layout.addWidget(self._roi_number)
        roi_layout.addStretch()
        roi_layout.addWidget(self._clear_btn)

        # LUT controls -----------------------------------------------------------

        # LUT slider
        self._clims = QLabeledRangeSlider(Qt.Orientation.Horizontal)
        self._clims.setStyleSheet(SS)
        self._clims.setHandleLabelPosition(
            QLabeledRangeSlider.LabelPosition.LabelsOnHandle
        )
        self._clims.setEdgeLabelMode(QLabeledRangeSlider.EdgeLabelMode.NoLabel)
        self._clims.setRange(0, 2**8)
        self._clims.valueChanged.connect(self._on_clims_changed)
        # auto contrast checkbox
        self._auto_clim = QPushButton("Auto")
        self._auto_clim.setCheckable(True)
        self._auto_clim.setChecked(True)
        self._auto_clim.toggled.connect(self._clims_auto)
        # reset view button
        self._reset_view = QPushButton()
        self._reset_view.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._reset_view.setToolTip("Reset View")
        self._reset_view.setIcon(icon(MDI6.fullscreen))
        self._reset_view.clicked.connect(self._reset)
        # bottom widget
        lut_wdg = QWidget()
        lut_wdg_layout = QHBoxLayout(lut_wdg)
        lut_wdg_layout.setContentsMargins(0, 0, 0, 0)
        lut_wdg_layout.addWidget(self._clims)
        lut_wdg_layout.addWidget(self._auto_clim)
        lut_wdg_layout.addWidget(self._reset_view)

        # Checkboxes --------------------------------------------------------------
        self.checkboxes = CheckBoxesGroup()
        self.checkboxes.valueChanged.connect(self._show_image)

        # Layout ------------------------------------------------------------------
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(roi_wdg)
        main_layout.addWidget(self._viewer)
        main_layout.addWidget(lut_wdg)
        main_layout.addWidget(self.checkboxes)

        # Connections -------------------------------------------------------------
        self._viewer.roiChanged.connect(self.roiChanged)

    def setData(
        self,
        image: np.ndarray,
        tubules: np.ndarray | None = None,
        nuclei: np.ndarray | None = None,
        brush_borders: np.ndarray | None = None,
        lumen: np.ndarray | None = None,
        glomerulus: np.ndarray | None = None,
    ) -> None:
        """Set the image data."""
        self._clear()
        if image is None:
            return

        # if len(image.shape) > 2:
        #     show_error_dialog(self, "Only 2D images are supported!")
        #     return

        self._clims.setRange(image.min(), image.max())

        self._viewer.update_image(
            image, tubules, nuclei, brush_borders, lumen, glomerulus
        )
        self._auto_clim.setChecked(True)

        with signals_blocked(self.checkboxes.image_cbox):
            self.checkboxes.image_cbox.setChecked(True)

        if tubules is None:
            self.checkboxes.tubules_cbox.setChecked(False)
        elif self._viewer.tubules is not None:
            self._viewer.tubules.visible = self.checkboxes.tubules_cbox.isChecked()

        if nuclei is None:
            self.checkboxes.nuclei_cbox.setChecked(False)
        elif self._viewer.nuclei is not None:
            self._viewer.nuclei.visible = self.checkboxes.nuclei_cbox.isChecked()

        if brush_borders is None:
            self.checkboxes.brush_borders_cbox.setChecked(False)
        elif self._viewer.brush_borders is not None:
            self._viewer.brush_borders.visible = (
                self.checkboxes.brush_borders_cbox.isChecked()
            )

        if lumen is None:
            self.checkboxes.lumen_cbox.setChecked(False)
        elif self._viewer.lumen is not None:
            self._viewer.lumen.visible = self.checkboxes.lumen_cbox.isChecked()

        if glomerulus is None:
            self.checkboxes.glomerulus_cbox.setChecked(False)
        elif self._viewer.glomerulus is not None:
            self._viewer.glomerulus.visible = (
                self.checkboxes.glomerulus_cbox.isChecked()
            )

    def data(self) -> np.ndarray | None:
        """Return the image data."""
        return self._viewer.image._data if self._viewer.image is not None else None

    def _on_clims_changed(self, range: tuple[float, float]) -> None:
        """Update the LUT range."""
        self._viewer.clims = range
        self._auto_clim.setChecked(False)

    def _clims_auto(self, state: bool) -> None:
        """Set the LUT range to auto."""
        self._viewer.clims = "auto" if state else self._clims.value()
        if self._viewer.image is not None:
            data = self._viewer.image._data
            with signals_blocked(self._clims):
                self._clims.setValue((data.min(), data.max()))

    def _reset(self) -> None:
        """Reset the view."""
        self._viewer.view.camera.set_range(margin=0)

    def _clear(self) -> None:
        """Clear the image."""
        for img in [
            self._viewer.image,
            self._viewer.tubules,
            self._viewer.nuclei,
            self._viewer.brush_borders,
            self._viewer.lumen,
            self._viewer.glomerulus,
        ]:
            if img is not None:
                img.parent = None
                img = None
        self._viewer.view.camera.set_range(margin=0)

    def _clear_highlight(self) -> None:
        """Clear the highlighted ROI."""
        if self._viewer.highlight_roi is not None:
            self._viewer.highlight_roi.parent = None
            self._viewer.highlight_roi = None
        self._roi_number.setText("")

    def _show_image(self, image_type: str, state: bool) -> None:
        """Show the labels."""
        self._clear_highlight()

        for img_type, img in [
            (IMAGE, self._viewer.image),
            (TUBULES, self._viewer.tubules),
            (NUCLEI, self._viewer.nuclei),
            (BORDERS, self._viewer.brush_borders),
            (LUMEN, self._viewer.lumen),
            (GLOMERULUS, self._viewer.glomerulus),
        ]:
            if img_type == image_type and img is not None:
                img.visible = state
    
    def _highlight_rois(self, roi: int) -> None:
        """Highlight the label set in the spinbox."""
        if roi == 0:
            return

        if self._viewer.image is None:
            # show_error_dialog(self, "No labels image to highlight.")
            return

        labels_data = self._viewer.tubules._data

        # Clear the previous highlight image if it exists
        if self._viewer.highlight_roi is not None:
            self._viewer.highlight_roi.parent = None
            self._viewer.highlight_roi = None

        # Create a mask for the label to highlight it
        highlight = np.zeros_like(labels_data, dtype=np.uint8)
        mask = labels_data == roi
        highlight[mask] = 255

        # Add the highlight image to the viewer
        self._viewer.highlight_roi = scene.visuals.Image(
            highlight,
            cmap="grays",
            clim=(0, 255),
            parent=self._viewer.view.scene,
        )
        self._viewer.highlight_roi.set_gl_state("additive", depth_test=False)
        self._viewer.highlight_roi.interactive = True
        self._viewer.highlight_roi.opacity = 0.9
        self._viewer.view.camera.set_range(margin=0)

        self._viewer.tubules.visible = False
        with signals_blocked(self.checkboxes.tubules_cbox):
            self.checkboxes.tubules_cbox.setChecked(False)


class _ImageCanvas(QWidget):
    """A Widget that displays an image."""

    roiChanged = Signal(object)

    def __init__(self, parent: Viewer):
        super().__init__(parent=parent)
        self._viewer = parent
        self._imcls = scene.visuals.Image
        self._clims: tuple[float, float] | Literal["auto"] = "auto"
        self._cmap: str = "grays"

        self._canvas = scene.SceneCanvas(keys="interactive", parent=self)
        self._canvas.events.mouse_move.connect(qthrottled(self._on_mouse_move, 60))
        self._canvas.events.mouse_press.connect(self._on_mouse_press)
        self.view = self._canvas.central_widget.add_view(camera="panzoom")
        self.view.camera.aspect = 1

        self._lbl = None

        # images
        self.image: scene.visuals.Image | None = None
        self.tubules: scene.visuals.Image | None = None
        self.nuclei: scene.visuals.Image | None = None
        self.brush_borders: scene.visuals.Image | None = None
        self.lumen: scene.visuals.Image | None = None
        self.glomerulus: scene.visuals.Image | None = None
        self.highlight_roi: scene.visuals.Image | None = None

        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().addWidget(self._canvas.native)

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    @property
    def clims(self) -> tuple[float, float] | Literal["auto"]:
        """Get the contrast limits of the image."""
        return self._clims

    @clims.setter
    def clims(self, clims: tuple[float, float] | Literal["auto"] = "auto") -> None:
        """Set the contrast limits of the image.

        Parameters
        ----------
        clims : tuple[float, float], or "auto"
            The contrast limits to set.
        """
        if self.image is not None:
            self.image.clim = clims
        self._clims = clims

    @property
    def cmap(self) -> str:
        """Get the colormap (lookup table) of the image."""
        return self._cmap

    @cmap.setter
    def cmap(self, cmap: str = "grays") -> None:
        """Set the colormap (lookup table) of the image.

        Parameters
        ----------
        cmap : str
            The colormap to use.
        """
        if self.image is not None:
            self.image.cmap = cmap
        self._cmap = cmap

    def update_image(
        self,
        image: np.ndarray,
        tubules: np.ndarray | None = None,
        nuclei: np.ndarray | None = None,
        brush_borders: np.ndarray | None = None,
        lumen: np.ndarray | None = None,
        glomerulus: np.ndarray | None = None,
    ) -> None:

        clim = (image.min(), image.max())
        self.image = self._imcls(
            image, cmap=self._cmap, clim=clim, parent=self.view.scene
        )
        self.image.set_gl_state("additive", depth_test=False)
        self.image.interactive = True
        self.view.camera.set_range(margin=0)

        if tubules is not None:
            self.tubules = self._imcls(
                tubules,
                cmap=self._labels_custom_cmap(tubules.max()),
                clim=(tubules.min(), tubules.max()),
                parent=self.view.scene,
            )
            self.tubules.set_gl_state("additive", depth_test=False)
            self.tubules.interactive = True
            self.tubules.visible = False
            self.tubules.opacity = 0.5

        for new_img, scene_img in [
            (nuclei, "nuclei"),
            (brush_borders, "brush_borders"),
            (lumen, "lumen"),
            (glomerulus, "glomerulus"),
        ]:
            if new_img is None:
                continue
            new_scene_img = self._imcls(
                new_img,
                cmap="grays",
                clim=(new_img.min(), new_img.max()),
                parent=self.view.scene,
            )
            new_scene_img.set_gl_state("additive", depth_test=False)
            new_scene_img.interactive = True
            new_scene_img.visible = False
            # new_scene_img.opacity = 0.9
            setattr(self, scene_img, new_scene_img)

    def _labels_custom_cmap(self, n_labels: int) -> Colormap:
        """Create a custom colormap for the labels."""
        colors = np.zeros((n_labels + 1, 4))
        colors[0] = [0, 0, 0, 1]  # Black for background (0)
        for i in range(1, n_labels + 1):
            colors[i] = [1, 1, 1, 1]  # White for all other labels
        return Colormap(colors)

    def _get_roi(self, event: SceneMouseEvent) -> None:
        """Update the pixel value when the mouse moves."""
        visual = self._canvas.visual_at(event.pos)
        image = self._find_image(visual)
        if image != self.tubules or image is None:
            self._viewer._roi_number.setText("")
            return
        tform = image.get_transform("canvas", "visual")
        px, py, *_ = (int(x) for x in tform.map(event.pos))
        return image._data[py, px]

    def _on_mouse_move(self, event: SceneMouseEvent) -> None:
        """Update the pixel value when the mouse moves."""
        roi = self._get_roi(event)
        roi = "" if roi == 0 else roi
        self._viewer._roi_number.setText(f"{roi}")

    def _on_mouse_press(self, event: SceneMouseEvent) -> None:
        """Highlight the selected ROI."""
        roi = self._get_roi(event)
        self.roiChanged.emit(roi)

    def _find_image(self, visual: scene.visuals.Visual) -> scene.visuals.Image | None:
        """Find the image visual in the visual tree."""
        if visual is None:
            return None
        if isinstance(visual, scene.visuals.Image):
            return visual
        for child in visual.children:
            image = self._find_image(child)
            if image is not None:
                return image
        return None


class CheckBoxesGroup(QGroupBox):
    """A group of checkboxes."""

    valueChanged = Signal(object, object)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        self.image_cbox = QCheckBox(IMAGE)
        self.tubules_cbox = QCheckBox(TUBULES)
        self.nuclei_cbox = QCheckBox(NUCLEI)
        self.brush_borders_cbox = QCheckBox(BORDERS)
        self.lumen_cbox = QCheckBox(LUMEN)
        self.glomerulus_cbox = QCheckBox(GLOMERULUS)

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)
        main_layout.addWidget(QLabel("Toggle to Show:"))
        main_layout.addWidget(self.image_cbox)
        main_layout.addWidget(self.tubules_cbox)
        main_layout.addWidget(self.nuclei_cbox)
        main_layout.addWidget(self.brush_borders_cbox)
        main_layout.addWidget(self.lumen_cbox)
        main_layout.addWidget(self.glomerulus_cbox)
        main_layout.addStretch()

        self.image_cbox.toggled.connect(self._emit_signal)
        self.tubules_cbox.toggled.connect(self._emit_signal)
        self.nuclei_cbox.toggled.connect(self._emit_signal)
        self.brush_borders_cbox.toggled.connect(self._emit_signal)
        self.lumen_cbox.toggled.connect(self._emit_signal)
        self.glomerulus_cbox.toggled.connect(self._emit_signal)

    def _emit_signal(self, state: bool) -> None:
        """Emit the valueChanged signal."""
        # get which checkbox was clicked
        checkbox = self.sender()
        self.valueChanged.emit(checkbox.text(), state)
