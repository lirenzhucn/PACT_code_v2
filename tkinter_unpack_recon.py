#!/usr/bin/env python

from Tkinter import *
from TkCommonDialog import CommonDialog, NumberEntry
from collections import OrderedDict
import argh

class ReconOpts:
    def __init__(self, entries):
        self.__dict__.update(entries)

RECON_OPTS_DICT = {
    # reconstructed image
    'ySize': 25.0,  # mm
    'xSize': 25.0,  # mm
    'yCenter': 0.0,  # mm
    'xCenter': 0.0,  # mm
    'iniAngle': 0.0,  # degrees
    'spacing': 0.25,  # mm
    # scanning geometry and parameters
    'R': 109.0,  # mm, scanning radius
    'fs': 20.0,  # MHz, sampling rate
    # other tunables
    'vm': 1.485,  # mm/us, speed of sound
}

class ConfigDialog(CommonDialog):
    def __init__(self, optsDict):
        self.result = optsDict
        # list of entry widgets
        self.entries = []
        self.keys = []
        CommonDialog.__init__(self)

    def body(self, master):
        rowIdx = 0
        self.keys = sorted(self.result.keys())
        for key in self.keys:
            val = self.result[key]
            Label(master, text=key).grid(row=rowIdx)
            self.entries.append(NumberEntry(master))
            self.entries[rowIdx].grid(row=rowIdx, column=1)
            self.entries[rowIdx].setVal(val)
            rowIdx = rowIdx + 1
        if self.entries:
            return self.entries[0]

    def apply(self):
        for i in xrange(len(self.keys)):
            key = self.keys[i]
            entry = self.entries[i]
            self.result[key] = entry.getVal()

def getReconOptsTk():
    cfgDlg = ConfigDialog(RECON_OPTS_DICT)
    cfgDlg.mainloop()
    return cfgDlg.result


import json

def main():
    opts = getReconOptsTk()
    print json.dumps(opts, indent=2)

if __name__ == '__main__':
    argh.dispatch_command(main)

