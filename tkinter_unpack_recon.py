#!/usr/bin/env python

from Tkinter import *
from TkCommonDialog import CommonDialog, NumberEntry
from ring_pact_reconstruction import *
from collections import OrderedDict
import argh

RECON_OPTS_DICT = {
    # reconstructed image
    'ySize': 25.0,  # mm
    'xSize': 25.0,  # mm
    'yCenter': 0.0,  # mm
    'xCenter': 0.0,  # mm
    'iniAngle': 225.0,  # degrees
    'spacing': 0.05,  # mm
    # scanning geometry and parameters
    'R': 25.0,  # mm, scanning radius
    'fs': 40.0,  # MHz, sampling rate
    # other tunables
    'vm': 1.510,  # mm/us, speed of sound
}

UNPACK_OPTS_DICT = {
    'BoardName': ['Board2004', 'Board9054'],
    'NumBoards': 2,
    'DataBlockSize': 1300,
    'PackSize': 8192,
    'NumDaqChnsBoard': 32,
    'TotFirings': 8,
    'NumElements': 512,
    'NumSegments': 1,
    'BadChannels': [21,22,23,24,85,86,87,88],
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
        resultDict = {}
        for i in xrange(len(self.keys)):
            key = self.keys[i]
            entry = self.entries[i]
            resultDict[key] = entry.getVal()
        self.result = Options(resultDict)

def getReconOptsTk():
    cfgDlg = ConfigDialog(RECON_OPTS_DICT)
    cfgDlg.mainloop()
    return cfgDlg.result


import json
import h5py
import pyfits
import os.path
import skimage.io._plugins.freeimage_plugin as fi

@argh.arg('input-file', type=str, help='input raw data file')
@argh.arg('-of', '--out-format', type=str, help='output format')
@argh.arg('-bp', '--bipolar', help='whether use bipolar recon.')
def reconstruct(input_file, out_format='fits', bipolar=False):
    print 'start reconstruction...'
    opts = getReconOptsTk()
    if bipolar:
        recon = Reconstruction2D(opts)
    else:
        recon = Reconstruction2DUnipolar(opts)
    f = h5py.File(input_file, 'r')
    paData = np.array(f['chndata_all'], order='F')
    f.close()
    reImg = recon.reconstruct(paData)
    (basename, ext) = os.path.splitext(input_file)
    if out_format == 'hdf5':
        outfile = basename + '_reImg.h5'
        print 'saving image data to ' + outfile
        f = h5py.File(outfile, 'w')
        f['reImg'] = reImg
        f.close()
    elif out_format == 'tiff':
        outfile = basename + '_reImg.tiff'
        print 'saving image data to ' + outfile
        imageList = [reImg[:,:,i] for i in xrange(reImg.shape[2])]
        fi.write_multipage(imageList, outfile)
    else: # including 'fits'
        outfile = basename + '_reImg.fits'
        print 'saving image data to ' + outfile
        hdu = pyfits.PrimaryHDU(reImg)
        hdu.writeto(outfile, clobber=True)


@argh.arg('-i', '--ind', type=int, help='index to process')
def unpack(src_dir, ind=-1):
    print 'start unpacking...'
    opts = Options(UNPACK_OPTS_DICT)
    opts.src_dir = src_dir
    opts.EXP_START = ind
    opts.EXP_END = ind
    opts.NUM_EXP = -1
    unpack = Unpack(opts)
    unpack.unpack()


if __name__ == '__main__':
    argh.dispatch_commands((unpack, reconstruct))

