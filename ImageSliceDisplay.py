#!/usr/bin/env python

from PyQt4.QtGui import *
from PyQt4.QtCore import Qt, pyqtSlot, SIGNAL
import numpy as np
from numpy import uint8

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.cm import register_cmap

from _kwave_cm import _kwave_data


# some constants
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


class ImageStat:
    '''Just like PIL's ImageStat.Stat class, this class provides lazily
    evaluated attributes about image statistics
    '''
    
    HIST_BIN_NUM = 256

    def __init__(self, imgData):
        self.imgData = imgData
        self._min = None
        self._max = None
        self._numSlices = None
        self._width = None
        self._height = None
        self._hist = None

    @property
    def extrema(self):
        return (self.min, self.max)

    @property
    def min(self):
        if self._min is None:
            self._min = np.amin(self.imgData)
        return self._min

    @property
    def max(self):
        if self._max is None:
            self._max = np.amax(self.imgData)
        return self._max

    @property
    def numSlices(self):
        if self._numSlices is None:
            self._numSlices = self.imgData.shape[2]
        return self._numSlices

    @property
    def imgSize(self):
        return (self.width, self.height)

    @property
    def width(self):
        if self._width is None:
            self._width = self.imgData.shape[1]
        return self._width

    @property
    def height(self):
        if self._height is None:
            self._height = self.imgData.shape[0]
        return self._height
    
    @property
    def hist(self):
        if self._hist is None:
            self._hist, junk = np.histogram(self.imgData, self.HIST_BIN_NUM)
        return self._hist


class ImageSliceDisplay(QWidget):
    
    def __init__(self, parent = None):
        QWidget.__init__(self, parent)
        self.dMin = 0.0
        self.dMax = 0.0
        self.imgData = None
        self.imgStat = None
        self.setupUi()
        
    def updateStatus(self):
        msg = '%d/%d; %d x %d' %\
         (self.mScSlice.value()+1, self.imgStat.numSlices,\
          self.imgStat.width, self.imgStat.height)
        self.mLbStatus.setText(msg)

    def setupUi(self):
        self.mLbStatus = QLabel(self)
        self.mLbDisplay = DoubleClickableLabel(self)
        self.mScSlice = QScrollBar(Qt.Horizontal, self)
        self.mScSlice.setPageStep(1)
        self.mScSlice.setMinimum(0)
        self.mScSlice.setMaximum(0)
        self.mScSlice.setSingleStep(1)
        # layout
        layout = QVBoxLayout(self)
        layout.addWidget(self.mLbStatus)
        layout.addWidget(self.mLbDisplay)
        layout.addWidget(self.mScSlice)
        # signal/slot pairs
        self.mScSlice.valueChanged.connect(self.onSliceChanged)
        self.connect(self.mLbDisplay, SIGNAL('doubleClicked()'),
                     self.onDisplayDoubleClicked)

    @pyqtSlot(int)
    def onSliceChanged(self, val):
        self.updateStatus()
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

    def setInput(self, imgData, cmapName):
        self.imgData = imgData
        self.imgStat = ImageStat(imgData)
        self.dMin, self.dMax = self.imgStat.extrema
        self.cmapName = cmapName
        # setup scroll bar
        self.mScSlice.setMaximum(imgData.shape[2] - 1)
        self.mScSlice.setValue(0)
        # setup Label size
        self.mLbDisplay.setFixedSize(imgData.shape[1], imgData.shape[0])
        # setup display image
        self.updateStatus()
        self.applyColormapStack()
        self.prepareQImage(0)
        self.update()
        
    def applyColormapStack(self):
        cmap = plt.get_cmap(self.cmapName)
        nSlices = self.imgData.shape[2]
        scaledData = (self.imgData - self.dMin) / (self.dMax - self.dMin)
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
    #app.setStyle('windows')
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

