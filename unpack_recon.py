#!/usr/bin/env python3

from tkinter import Label
from TkCommonDialog import CommonDialog, NumberEntry, ValCheckbutton
from ring_pact_reconstruction import Options
from ring_pact_reconstruction import Unpack, UnpackScan
from ring_pact_reconstruction import Reconstruction2D
from ring_pact_reconstruction import Reconstruction2DUnipolarHilbert
from ring_pact_reconstruction import Reconstruction2DUnipolarMultiview
from ring_pact_reconstruction import Reconstruction3D
from ring_pact_reconstruction import Reconstruction3DSingle
from ring_pact_reconstruction import ReconUtility
import numpy as np
from collections import OrderedDict

RECON_OPTS_DICT = OrderedDict([
    # reconstructed image
    ('ySize', 25.0),  # mm
    ('xSize', 25.0),  # mm
    ('yCenter', 0.0),  # mm
    ('xCenter', 0.0),  # mm
    ('iniAngle', 225.0),  # degrees
    ('spacing', 0.05),  # mm
    # scanning geometry and parameters
    ('R', 25.0),  # mm, scanning radius
    ('fs', 40.0),  # MHz, sampling rate
    # other tunables
    ('vm', 1.510),  # mm/us, speed of sound
    # preprocess flags
    ('exact', False),
    ('wiener', False),
    ('autoDelay', True),
    ('delay', 6.0),  # us
    ('method', 'bipolar'),  # method string
])

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
        self.keys = self.resultDict.keys()
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
            # valString = '%.3f' % (params[key],)
            valString = '{:.4f}'.format(params[key])
        elif key == 'spacing':
            # valString = '%.3f' % (params[key],)
            valString = '{:.3f}'.format(params[key])
        elif key == 'delay':
            valString = '{:.3f}'.format(params[key])
        elif key == 'method':
            valString = '{:}'.format(params[key])
            if params['exact']:
                valString = valString + '_exact'
            if params['wiener']:
                valString = valString + '_wiener'
        elif key == 'exact' or key == 'wiener':
            valString = ''
        else:
            # valString = '%s' % (params[key],)
            valString = '{:}'.format(params[key])
        filename = filename.replace(k, valString)
    return filename


import h5py
import pyfits
import os.path
import argh
from time import time
import json
import tifffile
import hdf5storage as h5


def sliceVolumeRead(folder):
    pass


def normalizeAndConvert(inData, dtype=None):
    if dtype is None:
        outData = inData / np.max(np.abs(inData))
        return outData
    if dtype == 'uint8':
        outData = (inData - np.min(inData)) / (np.max(inData) - np.min(inData))
        outData = outData * 255
        return outData.astype(np.uint8)
    elif dtype == 'int8':
        outData = inData / np.max(np.abs(inData))
        outData = outData * 127
        return outData.astype(np.int8)
    elif dtype == 'uint16':
        outData = (inData - np.min(inData)) / (np.max(inData) - np.min(inData))
        outData = outData * 65535
        return outData.astype(np.uint16)
    elif dtype == 'int16':
        outData = inData / np.max(np.abs(inData))
        outData = outData * 32767
        return outData.astype(np.int16)
    elif dtype == 'single':
        return inData.astype(np.float32)
    else:
        return inData


def readData(input_file, sliceNo):
    if os.path.isdir(input_file):
        fl = [os.path.join(input_file, f)
              for f in os.listdir(input_file)
              if os.path.isfile(os.path.join(input_file, f))]
        print('{:} is a directory.'.format(input_file))
        if sliceNo is not None:
            if sliceNo == 'mean':
                # read each file and average
                data0 = np.load(fl[0])
                data0 = np.zeros(data0.shape, dtype=np.float64, order='F')
                for f in fl:
                    data0 = data0 + np.load(f)
                paData = data0 / len(fl)
            else:
                sliceNo = int(sliceNo)
                paData = np.load(os.path.join(input_file,
                                              '{:06d}.npy'.format(sliceNo)))
                paData = paData.astype(np.float64)
        else:
            fl = sorted(fl)
            data0 = np.load(fl[0])
            paData = np.zeros((data0.shape[0], data0.shape[1], len(fl)),
                              dtype=np.float64, order='F')
            for i in range(len(fl)):
                f = fl[i]
                paData[:, :, i] = np.load(f)
                ReconUtility.updateProgress(i+1, len(fl))
    else:
        (basename, ext) = os.path.splitext(input_file)
        in_format = ext[1:]
        # read out data
        if in_format == 'h5':
            print('Reading data from {:}'.format(input_file))
            f = h5py.File(input_file, 'r')
            paData = np.array(f['data'], order='F')
            f.close()
            print('Done loading.')
        elif in_format == 'npy':
            paData = np.load(input_file)
        elif in_format == 'mat':
            paData = np.array(h5.loadmat(input_file)['data3'], order='F')
        else:
            print('input format %s not supported' % in_format)
            return
        if sliceNo is not None:
            if sliceNo == 'mean':
                print('reconstructing averaged')
                paData = np.array(np.mean(paData, axis=2), order='F')
            else:
                sliceNo = int(sliceNo)
                print('reconstructing slice #{:d}'.format(sliceNo))
                paData = np.copy(paData[:, :, sliceNo], order='F')
        paData = paData.astype(np.float64)
    return paData


@argh.arg('slice_no', type=int, help='Slice number for testing.')
@argh.arg('vm_range_str', type=str, help='Speed of sound range.')
@argh.arg('-vs', '--vm-step', type=float, help='Screen step size')
def screenspeed(input_file, output_file, opts_file, slice_no,
                vm_range_str, delay_range_str,
                vm_step=0.001, delay_step=0.025):
    '''
    Similar to reconstruct, with a required slice number and
    fixed unsigned int 8-bit data type (uint8)
    '''
    # process vm_range string first
    vm_range = sorted([float(v) for v in vm_range_str.split()])
    delay_range = sorted([float(d) for d in delay_range_str.split()])
    dtype = 'uint8'
    with open(opts_file) as fid:
        opts = json.load(fid, object_pairs_hook=Options)
        print(json.dumps(opts.__dict__, indent=2))
    if opts.method == 'bipolar':
        recon = Reconstruction2D(opts)
    elif opts.method == 'unipolar-hilbert':
        recon = Reconstruction2DUnipolarHilbert(opts)
    elif opts.method == 'unipolar-multiview':
        recon = Reconstruction2DUnipolarMultiview(opts)
    else:
        print('method {:} not supported'.format(opts.method))
        return
    paData = readData(input_file, slice_no)
    (basename, ext) = os.path.splitext(output_file)
    out_format = ext[1:]
    if out_format == '':
        output_file = output_file + '.tiff'
    if out_format != 'tiff':
        print('Output format must be tiff')
        return
    output_template = output_file
    for delay in list(np.arange(delay_range[0],
                                delay_range[1] + 0.5*delay_step,
                                delay_step)):
        recon.opts.delay = delay
        for vm in list(np.arange(vm_range[0], vm_range[1]+0.5*vm_step,
                                 vm_step)):
            print('testing vm = {:.4f} ...'.format(vm))
            recon.opts.vm = vm
            recon.initialized = False
            output_file = makeOutputFileName(output_template,
                                             recon.opts.__dict__)
            # dirname = os.path.dirname(input_file)
            # output_file = os.path.join(dirname, output_file)
            # reconstruction
            reImg = recon.reconstruct(paData)
            reImg = normalizeAndConvert(reImg, dtype)
            # make image viewable for TIFF viewers
            reImg = reImg.transpose((2, 0, 1))
            print('saving image data to ' + output_file)
            reImg = np.copy(reImg, order='C')
            tifffile.imsave(
                'slice_{:}_{:}'.format(slice_no, output_file), reImg)


def reconstruct_workhorse(input_file, output_file, opts,
                          sliceNo, timeit, dtype):
    if opts.method == 'bipolar':
        recon = Reconstruction2D(opts)
    elif opts.method == 'unipolar-hilbert':
        recon = Reconstruction2DUnipolarHilbert(opts)
    elif opts.method == 'unipolar-multiview':
        recon = Reconstruction2DUnipolarMultiview(opts)
    elif opts.method == 'bipolar-3d':
        recon = Reconstruction3D(opts)
    elif opts.method == 'bipolar-3d-single':
        recon = Reconstruction3DSingle(opts)
    else:
        print('method {:} not supported'.format(opts.method))
        return
    paData = readData(input_file, sliceNo)
    # reconstruction
    if timeit:
        startTime = time()
    reImg = recon.reconstruct(paData)
    if timeit:
        endTime = time()
        print('Reconstruction took %.2f seconds' % (endTime - startTime))
    output_file = makeOutputFileName(output_file, opts.__dict__)
    dirname = os.path.dirname(input_file)
    output_file = os.path.join(dirname, output_file)
    (basename, ext) = os.path.splitext(output_file)
    out_format = ext[1:]
    if out_format == 'h5':
        print('saving image data to ' + output_file)
        f = h5py.File(output_file, 'w')
        f['reImg'] = reImg
        f.close()
    elif out_format == 'tiff' or out_format == 'lsm':
        print('converting into {:}...'.format(dtype))
        reImg = normalizeAndConvert(reImg, dtype)
        # make image viewable for TIFF viewers
        reImg = reImg.transpose((2, 0, 1))
        print('saving image data to ' + output_file)
        reImg = np.copy(reImg, order='C')
        tifffile.imsave(output_file, reImg)
    elif out_format == 'npy':
        print('converting into {:}...'.format(dtype))
        reImg = normalizeAndConvert(reImg, dtype)
        print('saving image data to ' + output_file)
        np.save(output_file, reImg)
    elif out_format == 'mat':
        from scipy.io import savemat
        print('saving image data to ' + output_file)
        savemat(output_file, {'reImg': reImg})
    else:  # including 'fits'
        print('saving image data to ' + output_file)
        hdu = pyfits.PrimaryHDU(reImg)
        hdu.writeto(output_file, clobber=True)


@argh.arg('opts-file', type=str, help='option json file')
@argh.arg('input-file', type=str, help='input raw data file')
@argh.arg('output-file', type=str, help='output file name pattern')
@argh.arg('-s', '--slice-no', type=str, help='select a slice to reconstruct')
@argh.arg('-t', '--timeit', help='performance info')
@argh.arg('-d', '--dtype', type=str, help='data type for conversion')
def reconstruct(opts_file, input_file, output_file,
                slice_no=None, timeit=False, dtype='double'):
    print('start reconstruction...')
    # read options and replace unspecified items by defaults
    with open(opts_file) as fid:
        opts = json.load(fid, object_pairs_hook=Options)
        print(json.dumps(opts.__dict__, indent=2))
    reconstruct_workhorse(input_file, output_file, opts,
                          slice_no, timeit, dtype)


@argh.arg('input-file', type=str, help='input raw data file')
@argh.arg('output-file', type=str, help='output file name pattern')
@argh.arg('-t', '--timeit', help='performance info')
def reconstruct_gui(input_file, output_file, timeit=False):
    print('start reconstruction...')
    opts = ConfigDialog.getReconOptsTk()
    if opts is None:
        print('User cancelled reconstruction...')
        return
    reconstruct_workhorse(input_file, output_file, opts, timeit)


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


@argh.arg('src_dir', type=str, help='path to the source data directory')
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
    argh.dispatch_commands((unpack, unpack_scan, reconstruct,
                            reconstruct_gui, screenspeed))
