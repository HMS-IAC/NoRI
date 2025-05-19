from __future__ import annotations

from qtpy.QtWidgets import QWidget, QMenuBar

class MenuBar(QMenuBar):
    # Set the Menu
    def __init__(self, parent: QWidget | None = None):

        super().__init__(parent)
        self.files = self.addMenu('Files')
        self.opened = self.files.addAction('Open File')
        self.closed = self.files.addAction('Close File')

        self.opened.triggered.connect(self.open_text)
        self.closed.triggered.connect(self.close_text)


    def open_text(self):
        print('Opening')

    def close_text(self):
        print('Closing')