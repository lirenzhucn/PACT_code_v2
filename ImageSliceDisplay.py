#!/usr/bin/env python

from PyQt4.QtGui import *
from PyQt4.QtCore import Qt, pyqtSlot, SIGNAL
import numpy as np
from numpy import uint8

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.cm import register_cmap


_kwave_data = (
(0.240000, 0.240000, 0.332803),
(0.245000, 0.245000, 0.339697),
(0.250000, 0.250000, 0.346591),
(0.255000, 0.255000, 0.353485),
(0.260000, 0.260000, 0.360379),
(0.265000, 0.265000, 0.367273),
(0.270000, 0.270000, 0.374167),
(0.275000, 0.275000, 0.381061),
(0.280000, 0.280000, 0.387955),
(0.285000, 0.285000, 0.394848),
(0.290000, 0.290000, 0.401742),
(0.295000, 0.295000, 0.408636),
(0.300000, 0.300000, 0.415530),
(0.305000, 0.305000, 0.422424),
(0.310000, 0.310000, 0.429318),
(0.315000, 0.315000, 0.436212),
(0.320000, 0.320000, 0.443106),
(0.325000, 0.325000, 0.450000),
(0.330000, 0.331894, 0.455000),
(0.335000, 0.338788, 0.460000),
(0.340000, 0.345682, 0.465000),
(0.345000, 0.352576, 0.470000),
(0.350000, 0.359470, 0.475000),
(0.355000, 0.366364, 0.480000),
(0.360000, 0.373258, 0.485000),
(0.365000, 0.380152, 0.490000),
(0.370000, 0.387045, 0.495000),
(0.375000, 0.393939, 0.500000),
(0.380000, 0.400833, 0.505000),
(0.385000, 0.407727, 0.510000),
(0.390000, 0.414621, 0.515000),
(0.395000, 0.421515, 0.520000),
(0.400000, 0.428409, 0.525000),
(0.405000, 0.435303, 0.530000),
(0.410000, 0.442197, 0.535000),
(0.415000, 0.449091, 0.540000),
(0.420000, 0.455985, 0.545000),
(0.425000, 0.462879, 0.550000),
(0.430000, 0.469773, 0.555000),
(0.435000, 0.476667, 0.560000),
(0.440000, 0.483561, 0.565000),
(0.445000, 0.490455, 0.570000),
(0.450000, 0.497348, 0.575000),
(0.455000, 0.504242, 0.580000),
(0.460000, 0.511136, 0.585000),
(0.465000, 0.518030, 0.590000),
(0.470000, 0.524924, 0.595000),
(0.475000, 0.531818, 0.600000),
(0.480000, 0.538712, 0.605000),
(0.485000, 0.545606, 0.610000),
(0.490000, 0.552500, 0.615000),
(0.495000, 0.559394, 0.620000),
(0.500000, 0.566288, 0.625000),
(0.505000, 0.573182, 0.630000),
(0.510000, 0.580076, 0.635000),
(0.515000, 0.586970, 0.640000),
(0.520000, 0.593864, 0.645000),
(0.525000, 0.600758, 0.650000),
(0.530000, 0.607652, 0.655000),
(0.535000, 0.614545, 0.660000),
(0.540000, 0.621439, 0.665000),
(0.545000, 0.628333, 0.670000),
(0.550000, 0.635227, 0.675000),
(0.555000, 0.642121, 0.680000),
(0.560000, 0.649015, 0.685000),
(0.565000, 0.655909, 0.690000),
(0.570000, 0.662803, 0.695000),
(0.575000, 0.669697, 0.700000),
(0.580000, 0.676591, 0.705000),
(0.585000, 0.683485, 0.710000),
(0.590000, 0.690379, 0.715000),
(0.595000, 0.697273, 0.720000),
(0.600000, 0.704167, 0.725000),
(0.605000, 0.711061, 0.730000),
(0.610000, 0.717955, 0.735000),
(0.615000, 0.724848, 0.740000),
(0.620000, 0.731742, 0.745000),
(0.625000, 0.738636, 0.750000),
(0.630000, 0.745530, 0.755000),
(0.635000, 0.752424, 0.760000),
(0.640000, 0.759318, 0.765000),
(0.645000, 0.766212, 0.770000),
(0.650000, 0.773106, 0.775000),
(0.655000, 0.780000, 0.780000),
(0.662841, 0.785000, 0.785000),
(0.670682, 0.790000, 0.790000),
(0.678523, 0.795000, 0.795000),
(0.686364, 0.800000, 0.800000),
(0.694205, 0.805000, 0.805000),
(0.702045, 0.810000, 0.810000),
(0.709886, 0.815000, 0.815000),
(0.717727, 0.820000, 0.820000),
(0.725568, 0.825000, 0.825000),
(0.733409, 0.830000, 0.830000),
(0.741250, 0.835000, 0.835000),
(0.749091, 0.840000, 0.840000),
(0.756932, 0.845000, 0.845000),
(0.764773, 0.850000, 0.850000),
(0.772614, 0.855000, 0.855000),
(0.780455, 0.860000, 0.860000),
(0.788295, 0.865000, 0.865000),
(0.796136, 0.870000, 0.870000),
(0.803977, 0.875000, 0.875000),
(0.811818, 0.880000, 0.880000),
(0.819659, 0.885000, 0.885000),
(0.827500, 0.890000, 0.890000),
(0.835341, 0.895000, 0.895000),
(0.843182, 0.900000, 0.900000),
(0.851023, 0.905000, 0.905000),
(0.858864, 0.910000, 0.910000),
(0.866705, 0.915000, 0.915000),
(0.874545, 0.920000, 0.920000),
(0.882386, 0.925000, 0.925000),
(0.890227, 0.930000, 0.930000),
(0.898068, 0.935000, 0.935000),
(0.905909, 0.940000, 0.940000),
(0.913750, 0.945000, 0.945000),
(0.921591, 0.950000, 0.950000),
(0.929432, 0.955000, 0.955000),
(0.937273, 0.960000, 0.960000),
(0.945114, 0.965000, 0.965000),
(0.952955, 0.970000, 0.970000),
(0.960795, 0.975000, 0.975000),
(0.968636, 0.980000, 0.980000),
(0.976477, 0.985000, 0.985000),
(0.984318, 0.990000, 0.990000),
(0.992159, 0.995000, 0.995000),
(1.000000, 1.000000, 1.000000),
(1.000000, 1.000000, 1.000000),
(1.000000, 1.000000, 0.968750),
(1.000000, 1.000000, 0.937500),
(1.000000, 1.000000, 0.906250),
(1.000000, 1.000000, 0.875000),
(1.000000, 1.000000, 0.843750),
(1.000000, 1.000000, 0.812500),
(1.000000, 1.000000, 0.781250),
(1.000000, 1.000000, 0.750000),
(1.000000, 1.000000, 0.718750),
(1.000000, 1.000000, 0.687500),
(1.000000, 1.000000, 0.656250),
(1.000000, 1.000000, 0.625000),
(1.000000, 1.000000, 0.593750),
(1.000000, 1.000000, 0.562500),
(1.000000, 1.000000, 0.531250),
(1.000000, 1.000000, 0.500000),
(1.000000, 1.000000, 0.468750),
(1.000000, 1.000000, 0.437500),
(1.000000, 1.000000, 0.406250),
(1.000000, 1.000000, 0.375000),
(1.000000, 1.000000, 0.343750),
(1.000000, 1.000000, 0.312500),
(1.000000, 1.000000, 0.281250),
(1.000000, 1.000000, 0.250000),
(1.000000, 1.000000, 0.218750),
(1.000000, 1.000000, 0.187500),
(1.000000, 1.000000, 0.156250),
(1.000000, 1.000000, 0.125000),
(1.000000, 1.000000, 0.093750),
(1.000000, 1.000000, 0.062500),
(1.000000, 1.000000, 0.031250),
(1.000000, 1.000000, 0.000000),
(1.000000, 0.979167, 0.000000),
(1.000000, 0.958333, 0.000000),
(1.000000, 0.937500, 0.000000),
(1.000000, 0.916667, 0.000000),
(1.000000, 0.895833, 0.000000),
(1.000000, 0.875000, 0.000000),
(1.000000, 0.854167, 0.000000),
(1.000000, 0.833333, 0.000000),
(1.000000, 0.812500, 0.000000),
(1.000000, 0.791667, 0.000000),
(1.000000, 0.770833, 0.000000),
(1.000000, 0.750000, 0.000000),
(1.000000, 0.729167, 0.000000),
(1.000000, 0.708333, 0.000000),
(1.000000, 0.687500, 0.000000),
(1.000000, 0.666667, 0.000000),
(1.000000, 0.645833, 0.000000),
(1.000000, 0.625000, 0.000000),
(1.000000, 0.604167, 0.000000),
(1.000000, 0.583333, 0.000000),
(1.000000, 0.562500, 0.000000),
(1.000000, 0.541667, 0.000000),
(1.000000, 0.520833, 0.000000),
(1.000000, 0.500000, 0.000000),
(1.000000, 0.479167, 0.000000),
(1.000000, 0.458333, 0.000000),
(1.000000, 0.437500, 0.000000),
(1.000000, 0.416667, 0.000000),
(1.000000, 0.395833, 0.000000),
(1.000000, 0.375000, 0.000000),
(1.000000, 0.354167, 0.000000),
(1.000000, 0.333333, 0.000000),
(1.000000, 0.312500, 0.000000),
(1.000000, 0.291667, 0.000000),
(1.000000, 0.270833, 0.000000),
(1.000000, 0.250000, 0.000000),
(1.000000, 0.229167, 0.000000),
(1.000000, 0.208333, 0.000000),
(1.000000, 0.187500, 0.000000),
(1.000000, 0.166667, 0.000000),
(1.000000, 0.145833, 0.000000),
(1.000000, 0.125000, 0.000000),
(1.000000, 0.104167, 0.000000),
(1.000000, 0.083333, 0.000000),
(1.000000, 0.062500, 0.000000),
(1.000000, 0.041667, 0.000000),
(1.000000, 0.020833, 0.000000),
(1.000000, 0.000000, 0.000000),
(0.979167, 0.000000, 0.000000),
(0.958333, 0.000000, 0.000000),
(0.937500, 0.000000, 0.000000),
(0.916667, 0.000000, 0.000000),
(0.895833, 0.000000, 0.000000),
(0.875000, 0.000000, 0.000000),
(0.854167, 0.000000, 0.000000),
(0.833333, 0.000000, 0.000000),
(0.812500, 0.000000, 0.000000),
(0.791667, 0.000000, 0.000000),
(0.770833, 0.000000, 0.000000),
(0.750000, 0.000000, 0.000000),
(0.729167, 0.000000, 0.000000),
(0.708333, 0.000000, 0.000000),
(0.687500, 0.000000, 0.000000),
(0.666667, 0.000000, 0.000000),
(0.645833, 0.000000, 0.000000),
(0.625000, 0.000000, 0.000000),
(0.604167, 0.000000, 0.000000),
(0.583333, 0.000000, 0.000000),
(0.562500, 0.000000, 0.000000),
(0.541667, 0.000000, 0.000000),
(0.520833, 0.000000, 0.000000),
(0.500000, 0.000000, 0.000000),
(0.479167, 0.000000, 0.000000),
(0.458333, 0.000000, 0.000000),
(0.437500, 0.000000, 0.000000),
(0.416667, 0.000000, 0.000000),
(0.395833, 0.000000, 0.000000),
(0.375000, 0.000000, 0.000000),
(0.354167, 0.000000, 0.000000),
(0.333333, 0.000000, 0.000000),
(0.312500, 0.000000, 0.000000),
(0.291667, 0.000000, 0.000000),
(0.270833, 0.000000, 0.000000),
(0.250000, 0.000000, 0.000000),
(0.229167, 0.000000, 0.000000),
(0.208333, 0.000000, 0.000000),
(0.187500, 0.000000, 0.000000),
(0.166667, 0.000000, 0.000000),
(0.145833, 0.000000, 0.000000),
(0.125000, 0.000000, 0.000000),
(0.104167, 0.000000, 0.000000),
(0.083333, 0.000000, 0.000000),
(0.062500, 0.000000, 0.000000),
(0.041667, 0.000000, 0.000000),
(0.020833, 0.000000, 0.000000)
)

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
        #self.mLbDisplay = QLabel(self)
        self.mLbDisplay = DoubleClickableLabel(self)
        self.mScSlice = QScrollBar(Qt.Vertical, self)
        self.mScSlice.setMinimum(0)
        self.mScSlice.setMaximum(0)
        self.mScSlice.setSingleStep(1)
        # layout
        layout = QHBoxLayout(self)
        layout.addWidget(self.mLbDisplay)
        layout.addWidget(self.mScSlice)
        # singal/slot pairs
        self.mScSlice.valueChanged.connect(self.onSliceChanged)
        #self.mLbDisplay.doubleClicked.connect(self.onDisplayDoubleClicked)
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
    widget = ImageSliceDisplay()
    widget.setWindowTitle('Image Slice Display')
    widget.setInput(img, cmapName)
    widget.show()
    return app.exec_()


import skimage.io._plugins.freeimage_plugin as fi

if __name__ == '__main__':
    print 'Testing imshow function with a predefined input'
    imgList = fi.read_multipage('/home/liren/Documents/Project_data/PACT_data/2014-07-03-control/unpack/chndata_5_reImg.tiff')
    imgData = np.zeros((imgList[0].shape[0], imgList[0].shape[1], len(imgList)), dtype=np.double)
    for ind in xrange(len(imgList)):
        imgData[:,:,ind] = imgList[ind]
    imshow(imgData, 'kwave')

