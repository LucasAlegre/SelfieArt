from PyQt5 import QtCore, QtGui, QtWidgets
from selfie_art import SelfieArtCore
from PIL.ImageQt import ImageQt
from PIL import ImageColor
from ui import Ui_SelfieArt
from selfie_art import SelfieArtCore
import matplotlib.pyplot as plt


class MainWindow(Ui_SelfieArt):
    def __init__(self):
        super().__init__()
        self.selfie_art = SelfieArtCore()

    def setupUi(self, MW):
        super().setupUi(MW)

        self.selected_all = False

        self.applyButton.clicked.connect(self.applyStyle)
        self.selectImageButton.clicked.connect(self.readImage)
        self.selectStyleButton.clicked.connect(self.readStyle)
        self.selectAllButton.clicked.connect(self.selectAll)
        self.colorButton.clicked.connect(self.chooseColor)
        self.applyColorButton.clicked.connect(self.applyColor)
        self.saveButton.clicked.connect(self.saveResult)
        self.resetButton.clicked.connect(self.reset)

    @property
    def selectedSegments(self):
        return [i for i in range(19) if self.listWidget.item(i).checkState() == QtCore.Qt.Checked]

    def applyStyle(self):
        num_iterations = self.numIterationsBox.value()
        temperature = self.temperatureBox.value()
        overwrite = self.overCheckBox.isChecked()
        self.selfie_art.apply_style(self.selectedSegments, num_iterations=num_iterations, temperature=temperature, fast=self.fastCheckBox.isChecked(), over=overwrite)
        if self.selfie_art.image is not None:
            self.result.setPixmap(QtGui.QPixmap.fromImage(ImageQt(self.selfie_art.result_image)))

    def reset(self):
        self.selfie_art.reset()
        self.result.setPixmap(QtGui.QPixmap.fromImage(ImageQt(self.selfie_art.result_image)))

    def chooseColor(self):
        color = QtWidgets.QColorDialog.getColor()
        self.colorChosen.setStyleSheet("QWidget { background-color: %s}" % color.name())

    def applyColor(self):
        rgb = ImageColor.getcolor(self.colorChosen.palette().color(QtGui.QPalette.Background).name(), "RGB")
        temperature = self.temperatureBox.value()
        overwrite = self.overCheckBox.isChecked()
        self.selfie_art.apply_color(rgb, self.selectedSegments, temperature, over=overwrite)
        self.result.setPixmap(QtGui.QPixmap.fromImage(ImageQt(self.selfie_art.result_image)))

    def selectAll(self):
        state = QtCore.Qt.Unchecked if self.selected_all else QtCore.Qt.Checked
        self.selected_all = not self.selected_all
        for i in range(19):
            self.listWidget.item(i).setCheckState(state)
    
    def saveResult(self):
        filename = QtWidgets.QFileDialog.getSaveFileName(None, 'Save File')[0]
        if filename != '':
            self.selfie_art.save_result(filename)

    def readImage(self):
        fname = QtWidgets.QFileDialog.getOpenFileName(filter="Image files (*.jpg *.png)")[0]
        if fname != '':
            self.selfie_art.set_image(fname, run_face_parsing=True)
            self.result.resize(*self.selfie_art.image.size)
            self.img.setPixmap(QtGui.QPixmap.fromImage(ImageQt(self.selfie_art.image)))
    
    def readStyle(self):
        fname = QtWidgets.QFileDialog.getOpenFileName(filter="Image files (*.jpg *.png)")[0]
        if fname != '':
            self.selfie_art.set_style(fname)
            self.style.setPixmap(QtGui.QPixmap.fromImage(ImageQt(self.selfie_art.style_image)))


def main():
    import sys
    app = QtWidgets.QApplication(sys.argv)

    app.setStyle('Fusion')
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtGui.QColor(53,53,53))
    palette.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Base, QtGui.QColor(15,15,15))
    palette.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(53,53,53))
    palette.setColor(QtGui.QPalette.ToolTipBase, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.ToolTipText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Text, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.Button, QtGui.QColor(53,53,53))
    palette.setColor(QtGui.QPalette.ButtonText, QtCore.Qt.white)
    palette.setColor(QtGui.QPalette.BrightText, QtCore.Qt.red)
    palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(142,45,197).lighter())
    palette.setColor(QtGui.QPalette.HighlightedText, QtCore.Qt.black)
    app.setPalette(palette)

    icon = QtGui.QIcon()
    icon.addPixmap(QtGui.QPixmap("images/s-png-logo-transparent.png"), QtGui.QIcon.Selected, QtGui.QIcon.On)
    app.setWindowIcon(icon)

    SelfieArt = QtWidgets.QMainWindow()
    ui = MainWindow()
    ui.setupUi(SelfieArt)
    SelfieArt.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()