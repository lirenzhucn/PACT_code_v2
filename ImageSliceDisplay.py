#!/usr/bin/env python

from PyQt4.QtGui import *
from PyQt4.QtCore import Qt, pyqtSlot, SIGNAL
import numpy as np
from numpy import uint8

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.cm import register_cmap

from _kwave_cm import _kwave_data

kwave_cm = ListedColormap(_kwave_data, name='kwave')
register_cmap(name='kwave', cmap=kwave_cm)

class DoubleClickableLabel(QLabel):
    """A QLabel that sends out doubleClicked signal"""
    __pyqtSignals__ = ('doubleClicked()')
    def mouseDoubleClickEvent(self, event):
        self.emit(SIGNAL('doubleClicked()'))

class MinMaxDialog(QDialog):
    
    def __init__(self, dMin, dMax, parent = None):
        QDialog.__init__(self, parent)
        self.dMin = dMin
        self.dMax = dMax
        self.initUi()
        
    def initUi(self):
        minLabel = QLabel(self.tr('Min'))
        maxLabel = QLabel(self.tr('Max'))
        self.minEdit = QLineEdit(str(self.dMin))
        self.maxEdit = QLineEdit(str(self.dMax))
        self.minEdit.setValidator(QDoubleValidator())
        self.maxEdit.setValidator(QDoubleValidator())
        self.mBtnOK = QPushButton(self.tr('OK'))
        self.mBtnCancel = QPushButton(self.tr('Cancel'))
        self.mBtnOK.clicked.connect(self.accept)
        self.mBtnCancel.clicked.connect(self.reject)
        # layout
        gLayout = QGridLayout()
        gLayout.addWidget(minLabel, 1, 1)
        gLayout.addWidget(maxLabel, 2, 1)
        gLayout.addWidget(self.minEdit, 1, 2)
        gLayout.addWidget(self.maxEdit, 2, 2)
        hLayout = QHBoxLayout()
        hLayout.addWidget(self.mBtnOK)
        hLayout.addWidget(self.mBtnCancel)
        vLayout = QVBoxLayout()
        vLayout.addLayout(gLayout)
        vLayout.addLayout(hLayout)
        self.setLayout(vLayout)
        
    @pyqtSlot()
    def accept(self):
        self.dMin = float(self.minEdit.text())
        self.dMax = float(self.maxEdit.text())
        QDialog.accept(self)
        
    def getResults(self):
        return (self.dMin, self.dMax)


class ImageSliceDisplay(QWidget):
    
    def __init__(self, parent = None):
        QWidget.__init__(self, parent)
        self.dMin = 0.0
        self.dMax = 0.0
        self.data = None
        self.mLbDisplay = DoubleClickableLabel(self)
        self.mScSlice = QSlider(Qt.Vertical, self)
        self.mScSlice.setMinimum(0)
        self.mScSlice.setMaximum(0)
        self.mScSlice.setSingleStep(1)
        # layout
        layout = QHBoxLayout(self)
        layout.addWidget(self.mLbDisplay)
        layout.addWidget(self.mScSlice)
        # singal/slot pairs
        self.mScSlice.valueChanged.connect(self.onSliceChanged)
        self.connect(self.mLbDisplay, SIGNAL('doubleClicked()'),
                     self.onDisplayDoubleClicked)

    @pyqtSlot(int)
    def onSliceChanged(self, val):
        self.prepareQImage(val)
        self.update()

    @pyqtSlot()
    def onDisplayDoubleClicked(self):
        mmDialog = MinMaxDialog(self.dMin, self.dMax, self)
        ret = mmDialog.exec_()
        if ret == 1:
            (self.dMin, self.dMax) = mmDialog.getResults()
            self.applyColormapStack()
            self.prepareQImage(self.mScSlice.value())
            self.update()

    def setInput(self, data, cmapName):
        self.data = data
        self.dMin = np.amin(data)
        self.dMax = np.amax(data)
        self.cmapName = cmapName
        # setup scroll bar
        self.mScSlice.setMaximum(data.shape[2] - 1)
        self.mScSlice.setValue(0)
        # setup Label size
        self.mLbDisplay.setFixedSize(data.shape[1], data.shape[0])
        # setup display image
        self.applyColormapStack()
        self.prepareQImage(0)
        self.update()
        
    def applyColormapStack(self):
        cmap = plt.get_cmap(self.cmapName)
        nSlices = self.data.shape[2]
        scaledData = (self.data - self.dMin) / (self.dMax - self.dMin)
        scaledData[scaledData < 0.0] = 0
        scaledData[scaledData > 1.0] = 1.0
        self.rgbaStack = [None] * nSlices
        for ind in xrange(nSlices):
            rgbaImg = cmap(scaledData[:,:,ind], bytes=True)
            self.rgbaStack[ind] = np.zeros(rgbaImg.shape, dtype=np.uint8)
            self.rgbaStack[ind][:,:,0] = rgbaImg[:,:,2]
            self.rgbaStack[ind][:,:,1] = rgbaImg[:,:,1]
            self.rgbaStack[ind][:,:,2] = rgbaImg[:,:,0]
            self.rgbaStack[ind][:,:,3] = 255

    def prepareQImage(self, ind):
        rgbaImg = self.rgbaStack[ind]
        assert(rgbaImg is not None)
        self.img = QImage(rgbaImg.tostring(order='C'),\
                          rgbaImg.shape[1], rgbaImg.shape[0],\
                          QImage.Format_RGB32)
        pix = QPixmap.fromImage(self.img)
        self.mLbDisplay.setPixmap(pix.scaled(self.mLbDisplay.size(),\
                                             Qt.KeepAspectRatio,\
                                             Qt.SmoothTransformation))


def imshow(img, cmapName='gray'):
    '''imshow:
    Display a 2D or 3D image with a matplotlib predefined colormap
    in the Qt-powered ImageSliceDisplay widget.
    '''
    app = QApplication([])
    app.setStyle('windows')
    widget = ImageSliceDisplay()
    widget.setWindowTitle('Image Slice Display')
    widget.setInput(img, cmapName)
    widget.show()
    return app.exec_()


import skimage.io._plugins.freeimage_plugin as fi
import argh

@argh.arg('input_file', type=str, help='path to input image file')
@argh.arg('cm_name', type=str, help='colormap name')
def main(input_file, cm_name):
    print 'Testing imshow function with a predefined input'
    imgList = fi.read_multipage(input_file)
    imgData = np.zeros((imgList[0].shape[0], imgList[0].shape[1],\
                        len(imgList)), dtype=np.double)
    for ind in xrange(len(imgList)):
        imgData[:,:,ind] = imgList[ind]
    imshow(imgData, cm_name)

if __name__ == '__main__':
    argh.dispatch_command(main)

