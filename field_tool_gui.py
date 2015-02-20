#!/usr/bin/env python3

"""
This is a script specifically designed to be used during experiment.
"""


from PyQt4 import QtCore, QtGui
from queue import Queue
from ring_pact_reconstruction import Options


class LogListener(QtCore.QThread):

    """
    A Qt thread that listens to a string queue for 'messages'
    and transmit the strings through a Qt signal, so they can
    be picked up and further processed by other Qt objects.
    """

    mysignal = QtCore.pyqtSignal(str)

    def __init__(self, queue):
        QtCore.QThread.__init__(self)
        self.queue = queue

    def run(self):
        while True:
            text = self.queue.get()
            self.mysignal.emit(text)


class OutLogger:

    """
    A stdout substitute to intercept and redirect system output to a
    LogListener object, which later emit a Qt signal to other Qt objects.
    """

    def __init__(self):
        self.queue = Queue()
        self.listener = LogListener(self.queue)

    def write(self, m):
        self.queue.put(m)

    def flush(self):
        pass


from ring_pact_reconstruction import Unpack


class UnpackThread(QtCore.QThread):
    """ Worker thread for unpack data """

    UNPACK_OPTS_DICT = {
        'BoardName': ['Board2004', 'Board9054'],
        'NumBoards': 2,
        'DataBlockSize': 1300,
        'PackSize': 8192,
        'NumDaqChnsBoard': 32,
        'TotFirings': 8,
        'NumElements': 512,
        'NumSegments': 1,
        'BadChannels': [21, 22, 23, 24, 85, 86, 87, 88],
    }

    def __init__(self, srcDir='', ind=-1):
        # super().__init__(self)
        QtCore.QThread.__init__(self)
        self.opts = Options(self.UNPACK_OPTS_DICT)
        self.opts.src_dir = srcDir
        self.opts.EXP_START = ind
        self.opts.EXP_END = ind
        self.opts.NUM_EXP = -1

    def setParameters(self, srcDir, ind):
        self.opts.src_dir = srcDir
        self.opts.EXP_START = ind
        self.opts.EXP_END = ind
        self.opts.NUM_EXP = -1

    def run(self):
        self.unpack = Unpack(self.opts)
        self.unpack.unpack()


class ReconstructThread(QtCore.QThread):
    """Worker thread for reconstruct image"""
    pass

from ui_fieldToolDialog import Ui_FieldToolDialog


class MainDialog(QtGui.QDialog, Ui_FieldToolDialog):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.configUi()
        self.setupCommunication()
        self.setupThreads()
        self.unpackOpts = {}
        self.reconOpts = {}

    def configUi(self):
        """
        Configure GUI components here
        """
        self.mEdLog.setWordWrapMode(QtGui.QTextOption.WrapAnywhere)

    def setupCommunication(self):
        """
        Configure GUI components and signals/slots
        """
        self.mBtnClearLog.clicked.connect(self.onClearLog)
        self.mBtnChooseDataFolder.clicked.connect(self.onChooseDataFolder)
        self.mBtnClose.clicked.connect(self.onClose)
        self.mBtnUnpack.clicked.connect(self.onUnpack)

    def setupThreads(self):
        # create threads
        self.unpackThread = UnpackThread()
        # connect threads' signals to slots
        self.unpackThread.terminated.connect(self.onWorkerTerminated)
        self.unpackThread.finished.connect(self.onWorkerDone)

    def setupLogger(self, logger):
        """
        Hook a OutLogger object with the dialog.
        The dialog will handle the OutLogger's only signal 'mysignal'
        """
        logger.listener.mysignal.connect(self.logText)
        logger.listener.start()

    @QtCore.pyqtSlot(str)
    def logText(self, text):
        self.mEdLog.moveCursor(QtGui.QTextCursor.End)
        self.mEdLog.insertPlainText(text)

    @QtCore.pyqtSlot()
    def onClose(self):
        self.done(0)

    @QtCore.pyqtSlot()
    def onClearLog(self):
        self.mEdLog.clear()

    @QtCore.pyqtSlot()
    def onChooseDataFolder(self):
        """Called when Choose button is pressed"""
        pathStr = str(self.mEdDataFolder.text())
        pathStr = QtGui.QFileDialog.getExistingDirectory(
            self, 'Data folder', pathStr)
        self.mEdDataFolder.setText(pathStr)

    @QtCore.pyqtSlot()
    def onUnpack(self):
        """Unpack button clicked"""
        srcDir = self.mEdDataFolder.text()
        ind = int(self.mEdIndex.text())
        self.unpackThread.setParameters(srcDir, ind)
        self.unpackThread.start()

    @QtCore.pyqtSlot()
    def onWorkerDone(self):
        print('Worker thread finished.')

    @QtCore.pyqtSlot()
    def onWorkerTerminated(self):
        print('Worker thread terminated.')


if __name__ == '__main__':
    import sys
    app = QtGui.QApplication(sys.argv)
    app.setFont(QtGui.QFont('Arial', 12))
    mainDialog = MainDialog()
    # redirect output to Log TextEdit
    outLogger = OutLogger()
    mainDialog.setupLogger(outLogger)
    sys.stdout = outLogger
    ret = mainDialog.exec_()
