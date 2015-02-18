#!/usr/bin/env python3

"""
This is a script specifically designed to be used during experiment.
"""


from PyQt4 import QtCore, QtGui
from queue import Queue


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


class UnpackThread(QtCore.QThread):
    """
    Worker thread for unpack data
    """
    pass

from ui_fieldToolDialog import Ui_FieldToolDialog


class MainDialog(QtGui.QDialog, Ui_FieldToolDialog):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.configUi()
        self.setupCommunication()
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


if __name__ == '__main__':
    import sys
    app = QtGui.QApplication(sys.argv)
    app.setFont(QtGui.QFont('Arial', 12))
    mainDialog = MainDialog()
    # redirect output to Log TextEdit
    outLogger = OutLogger()
    mainDialog.setupLogger(outLogger)
    sys.stdout = outLogger
    print('Done initialization.')
    ret = mainDialog.exec_()
