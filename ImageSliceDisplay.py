#!/usr/bin/env python

from PyQt4.QtGui import *
from PyQt4.QtCore import Qt, pyqtSlot, SIGNAL
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.cm import register_cmap

from _kwave_cm import _kwave_data
from matplotlib.pyplot import hist


# some constants
kwave_cm = ListedColormap(_kwave_data, name='kwave')
register_cmap(name='kwave', cmap=kwave_cm)


class DoubleClickableLabel(QLabel):
    """A QLabel that sends out doubleClicked signal"""
    __pyqtSignals__ = ('doubleClicked()')
    def mouseDoubleClickEvent(self, event):
        self.emit(SIGNAL('doubleClicked()'))

class MinMaxDialog(QDialog):
    
    __pyqtSignals__ = ('minMaxChanged()')
    
    def __init__(self, dMin, dMax, imgStat, parent = None):
        QDialog.__init__(self, parent)
        self.dMin = dMin
        self.dMax = dMax
        self.imgStat = imgStat
        self.setupUi()
        # signals and slots
        self.mBtnClose.clicked.connect(self.accept)
        self.mScMin.valueChanged.connect(self.minMaxChange)
        self.mScMax.valueChanged.connect(self.minMaxChange)
        
    def setupHistogram(self):
        hist = list(self.imgStat.hist)
        hist = [float(v)/max(hist) for v in hist]
        width = len(hist)
        height = len(hist)/2
        self.mLbHist = QLabel(self)
        self.mLbHist.setFixedSize(width, height)
        self.mPixmapHist = QPixmap(width, height)
        self.mPixmapHist.fill()
        qp = QPainter()
        qp.begin(self.mPixmapHist)
        qp.setPen(QColor(100, 100, 100))
        for ind in xrange(len(hist)):
            qp.drawLine(ind,height,ind,(1-hist[ind])*height)
        qp.end()
        #self.mLbHist.setPixmap(self.mPixmapHist)
        self.drawHistLabel()
        
    def drawHistLabel(self):
        width = self.mPixmapHist.width()
        height = self.mPixmapHist.height()
        lp = int((self.dMin-self.imgStat.min)/self.imgStat.range*width)
        rp = int((self.dMax-self.imgStat.min)/self.imgStat.range*width)
        pixmap = QPixmap(width, height)
        qp = QPainter()
        qp.begin(pixmap)
        qp.drawPixmap(0, 0, self.mPixmapHist)
        qp.setPen(QColor(0, 0, 0))
        qp.drawLine(lp,height,rp,0)
        qp.end()
        self.mLbHist.setPixmap(pixmap)
        
    def setupUi(self):
        self.setWindowTitle('Min & Max')
        # histogram
        self.setupHistogram()
        # sliders
        self.mScMin = QScrollBar(Qt.Horizontal, self)
        self.mScMin.setPageStep(1)
        self.mScMin.setSingleStep(1)
        self.mScMin.setMinimum(0)
        self.mScMin.setMaximum(100)
        self.mScMin.setValue(0)
        lbMin = QLabel('Minimum', self)
        lbMin.setAlignment(Qt.AlignCenter)
        self.mScMax = QScrollBar(Qt.Horizontal, self)
        self.mScMax.setPageStep(1)
        self.mScMax.setSingleStep(1)
        self.mScMax.setMinimum(0)
        self.mScMax.setMaximum(100)
        self.mScMax.setValue(100)
        lbMax = QLabel('Maximum', self)
        lbMax.setAlignment(Qt.AlignCenter)
        # buttons
        self.mBtnClose = QPushButton('Close', self)
        # layout
        layout = QVBoxLayout(self)
        layout.addWidget(self.mLbHist)
        layout.addWidget(self.mScMin)
        layout.addWidget(lbMin)
        layout.addWidget(self.mScMax)
        layout.addWidget(lbMax)
        layout.addWidget(self.mBtnClose)
        
    @pyqtSlot(int)
    def minMaxChange(self, int):
        self.dMin = self.mScMin.value()/100.0*self.imgStat.range +\
            self.imgStat.min
        self.dMax = self.mScMax.value()/100.0*self.imgStat.range +\
            self.imgStat.min
        self.drawHistLabel()
        self.mLbHist.update()
        self.emit(SIGNAL('minMaxChanged()'))
        
    @property
    def results(self):
        return (self.dMin, self.dMax)


class ImageStat:
    '''Just like PIL's ImageStat.Stat class, this class provides lazily
    evaluated attributes about image statistics
    '''
    
    HIST_BIN_NUM = 128

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
    
    @property
    def range(self):
        return self.max - self.min


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
    def minMaxChange(self):
        self.dMin, self.dMax = self.mmDialog.results
        self.prepareQImage(self.mScSlice.value())
        self.update()

    @pyqtSlot()
    def onDisplayDoubleClicked(self):
        self.mmDialog = MinMaxDialog\
            (self.dMin, self.dMax, self.imgStat, self)
        self.connect(self.mmDialog, SIGNAL('minMaxChanged()'),
                     self.minMaxChange)
        self.mmDialog.exec_()
        self.disconnect(self.mmDialog, SIGNAL('minMaxChanged()'),
                        self.minMaxChange)

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
        self.prepareQImage(0)
        self.update()

    def prepareQImage(self, ind):
#         rgbaImg = self.rgbaStack[ind]
#         assert(rgbaImg is not None)
        img = self.imgData[:,:,ind]
        scaledImg = (img - self.dMin) / (self.dMax - self.dMin)
        scaledImg[scaledImg < 0.0] = 0.0
        scaledImg[scaledImg > 1.0] = 1.0
        cmap = plt.get_cmap(self.cmapName)
        rgbaImg_temp = cmap(scaledImg, bytes=True)
        rgbaImg = np.zeros(rgbaImg_temp.shape, dtype=np.uint8)
        rgbaImg[:,:,0] = rgbaImg_temp[:,:,2]
        rgbaImg[:,:,1] = rgbaImg_temp[:,:,1]
        rgbaImg[:,:,2] = rgbaImg_temp[:,:,0]
        rgbaImg[:,:,3] = 255
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
    #app.setStyle('plastique')
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

