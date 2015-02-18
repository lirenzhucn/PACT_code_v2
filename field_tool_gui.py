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


from ui_fieldToolDialog import Ui_FieldToolDialog


class MainDialog(QtGui.QDialog, Ui_FieldToolDialog):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.configUi()
        self.setupCommunication()

    def configUi(self):
        """
        Configure GUI components here
        """
        pass

    def setupCommunication(self):
        """
        Configure GUI components and signals/slots
        """
        pass

    def setupLogger(self, logger):
        """
        Hook a OutLogger object with the dialog.
        The dialog will handle the OutLogger's only signal 'mysignal'
        """
        # logger.listener.mysignal.connect(self.logText)
        # logger.listener.start()
        pass


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
