#!/usr/bin/env python3

from tkinter import Label
from TkCommonDialog import CommonDialog, NumberEntry, ValCheckbutton
from ring_pact_reconstruction import Options
from ring_pact_reconstruction import Unpack, UnpackScan
from ring_pact_reconstruction import Reconstruction2D
from ring_pact_reconstruction import Reconstruction2DUnipolar_v2
import numpy as np

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
    # preprocess flags
    'exact': False,
    'wiener': False,
    'autoDelay': True,
    'delay': 6.0,  # us
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
    'BadChannels': [21, 22, 23, 24, 85, 86, 87, 88],
}


class ConfigDialog(CommonDialog):

    def __init__(self, optsDict):
        self.resultDict = optsDict
        self.result = None
        # list of entry widgets
        self.entries = []
        self.keys = []
        CommonDialog.__init__(self)

    def body(self, master):
        rowIdx = 0
        self.keys = sorted(self.resultDict.keys())
        for key in self.keys:
            val = self.resultDict[key]
            if isinstance(val, bool):
                self.entries.append(ValCheckbutton(master, text=key))
            else:  # assuming everything is number
                Label(master, text=key).grid(row=rowIdx)
                self.entries.append(NumberEntry(master))
            self.entries[rowIdx].grid(row=rowIdx, column=1)
            self.entries[rowIdx].setVal(val)
            rowIdx = rowIdx + 1
        if self.entries:
            return self.entries[0]

    def apply(self):
        resultDict = {}
        for i in range(len(self.keys)):
            key = self.keys[i]
            entry = self.entries[i]
            resultDict[key] = entry.getVal()
        self.result = Options(resultDict)

    @staticmethod
    def getReconOptsTk():
        cfgDlg = ConfigDialog(RECON_OPTS_DICT)
        cfgDlg.mainloop()
        return cfgDlg.result


def makeOutputFileName(pattern, params):
    filename = pattern
    for key in params.keys():
        k = r'{' + key + r'}'
        if key == 'vm':
            valString = '%.3f' % (params[key],)
        elif key == 'spacing':
            valString = '%.3f' % (params[key],)
        else:
            valString = '%.1f' % (params[key],)
        filename = filename.replace(k, valString)
    return filename


import h5py
import pyfits
import os.path
import skimage.io._plugins.freeimage_plugin as fi
import argh
from time import time


@argh.arg('input-file', type=str, help='input raw data file')
@argh.arg('out-file', type=str, help='output file name pattern')
@argh.arg('-8', '--eight-bit',
          help='save 8-bit version for display too.')
@argh.arg('-bp', '--bipolar', help='whether use bipolar recon.')
@argh.arg('-t', '--timeit', help='performance info')
def reconstruct(input_file, out_file,
                eight_bit=False, bipolar=False, timeit=False):
    print('start reconstruction...')
    opts = ConfigDialog.getReconOptsTk()
    if opts is None:
        print('User cancelled reconstruction...')
        return
    if bipolar:
        recon = Reconstruction2D(opts)
    else:
        recon = Reconstruction2DUnipolar_v2(opts)
    (basename, ext) = os.path.splitext(input_file)
    in_format = ext[1:]
    # read out data
    if in_format == 'h5':
        f = h5py.File(input_file, 'r')
        paData = np.array(f['chndata_all'], order='F')
        f.close()
    elif in_format == 'npy':
        paData = np.load(input_file)
    # reconstruction
    if timeit:
        startTime = time()
    reImg = recon.reconstruct(paData)
    if timeit:
        endTime = time()
        print('Reconstruction took %.2f seconds' % (endTime - startTime))
    out_file = makeOutputFileName(out_file, opts.__dict__)
    dirname = os.path.dirname(input_file)
    out_file = os.path.join(dirname, out_file)
    (basename, ext) = os.path.splitext(out_file)
    out_format = ext[1:]
    if out_format == 'hdf5':
        print('saving image data to ' + out_file)
        f = h5py.File(out_file, 'w')
        f['reImg'] = reImg
        f.close()
    elif out_format == 'tiff':
        print('saving image data to ' + out_file)
        imageList = [reImg[:, :, i] for i in range(reImg.shape[2])]
        fi.write_multipage(imageList, out_file)
    elif out_format == 'npy':
        print('saving image data to ' + out_file)
        np.save(out_file, reImg)
    else:  # including 'fits'
        print('saving image data to ' + out_file)
        hdu = pyfits.PrimaryHDU(reImg)
        hdu.writeto(out_file, clobber=True)
    # save 8-bit image for display if needed
    if not eight_bit:
        return
    reImg8bit = ((reImg - np.amin(reImg)) /
                 (np.amax(reImg) - np.amin(reImg)) * 255.0).\
        astype(np.uint8)
    (basename, ext) = os.path.splitext(out_file)
    out_file = basename + '_8bit' + ext
    if out_format == 'hdf5':
        print('saving image data to ' + out_file)
        f = h5py.File(out_file, 'w')
        f['reImg'] = reImg8bit
        f.close()
    elif out_format == 'tiff':
        print('saving image data to ' + out_file)
        imageList = [reImg8bit[:, :, i] for i in range(reImg.shape[2])]
        fi.write_multipage(imageList, out_file)
    elif out_format == 'npy':
        print('saving image data to ' + out_file)
        np.save(out_file, reImg8bit)
    else:  # including 'fits'
        print('saving image data to ' + out_file)
        hdu = pyfits.PrimaryHDU(reImg8bit)
        hdu.writeto(out_file, clobber=True)


@argh.arg('-i', '--ind', type=int, help='index to process')
def unpack(src_dir, ind=-1):
    print('start unpacking...')
    opts = Options(UNPACK_OPTS_DICT)
    opts.src_dir = src_dir
    opts.EXP_START = ind
    opts.EXP_END = ind
    opts.NUM_EXP = -1
    unpack = Unpack(opts)
    unpack.unpack()


@argh.arg('src_dir', type=str,
          help='path to the source data directory')
@argh.arg('min_ind', type=int, help='starting index')
@argh.arg('max_ind', type=int, help='ending index')
def unpack_scan(src_dir, min_ind, max_ind):
    print('start unpacking...')
    opts = Options(UNPACK_OPTS_DICT)
    opts.src_dir = src_dir
    opts.EXP_START = min_ind
    opts.EXP_END = max_ind
    opts.NUM_EXP = -1
    unpack = UnpackScan(opts)
    unpack.unpack()


if __name__ == '__main__':
    argh.dispatch_commands((unpack, unpack_scan, reconstruct))
