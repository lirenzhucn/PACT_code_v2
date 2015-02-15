'''
This is a library that provide reconstruction functionalities
This library has been modified to work with Python 3
'''

from math import floor
import re
import numpy as np
import scipy.signal as spsig
import scipy.ndimage as spnd
import h5py
from os.path import expanduser, normpath, join
from os import listdir, rename
import sys

from ring_pact_speedup import daq_loop, generateChanMap
from ring_pact_speedup import recon_loop, find_index_map_and_angular_weight
from ring_pact_speedup import backproject_loop
from preprocess import subfunc_wiener, subfunc_exact


class Options:

    def __init__(self, entries):
        self.__dict__.update(entries)


class UnpackUtility:
    '''Provide a series of statistic methods for Unpack classes'''

    @staticmethod
    def sizeOfAxis(x, ind):
        return x.shape[ind] if x is not None else 0

    @staticmethod
    def readBinFile(filePath, dtype, packSize, totFirings, numExpr):
        try:
            f = open(filePath)
            tempData = np.fromfile(f, dtype=dtype)
            f.close()
            tempData = tempData.reshape((6*totFirings*numExpr, packSize)).T
            tempData = np.copy(tempData, order='C')
            return tempData
        except:
            return None

    @staticmethod
    def saveChnData(chnData, chnDataAll, destDir, ind):
        fileName = 'chndata_%d.h5' % (ind)
        outputPath = join(destDir, fileName)
        print('Saving data to ' + outputPath)
        f = h5py.File(outputPath, 'w')
        f['chndata'] = chnData
        f['chndata_all'] = chnDataAll
        f.close()

    @staticmethod
    def renameUnindexedFile(srcDir):
        print('Renaming unindexed raw data files in %s' % (srcDir))
        pTarget = re.compile(r'Board([0-9]+)Experiment([0-9]+)' +
                             r'TotalFiring([0-9]+)_Pack.bin')
        pExisting = re.compile(r'Board([0-9]+)Experiment([0-9]+)' +
                               r'TotalFiring([0-9]+)_Pack_([0-9]+).bin')
        # find unindexed bin file and the max index
        targetFileList = []
        indexList = []
        for fileName in listdir(srcDir):
            if pTarget.match(fileName) is not None:
                targetFileList.append(fileName)
            else:
                matchExisting = pExisting.match(fileName)
                if matchExisting is not None:
                    indexList.append(int(matchExisting.group(4)))
        if not targetFileList:
            print('No unindexed file found!')
            return -1
        # target index is max index +1
        if not indexList:
            renameIndex = 1
        else:
            renameIndex = max(indexList) + 1
        for fileName in targetFileList:
            srcFilePath = join(srcDir, fileName)
            destFilePath = '%s_%d.bin' % (srcFilePath[:-4], renameIndex)
            print(srcFilePath)
            print('\t->' + destFilePath)
            rename(srcFilePath, destFilePath)
        return renameIndex


class UnpackScan:
    '''Extractor for averaged/scanned data set'''

    def __init__(self, opts):
        assert(isinstance(opts, Options))
        self.opts = opts
        if not hasattr(self.opts, 'dest_ext'):
            self.opts.dest_ext = 'unpack'
        if not hasattr(self.opts, 'dtype'):
            self.opts.dtype = '<u4'
        # normalize and expand the paths
        self.opts.src_dir = normpath(expanduser(self.opts.src_dir))
        self.opts.dest_dir = normpath(expanduser(join(self.opts.src_dir,
                                                      self.opts.dest_ext)))

    def readChannelData(self):
        # variable extraction
        startInd = self.opts.EXP_START
        endInd = self.opts.EXP_END
        srcDir = self.opts.src_dir
        # destDir = self.opts.dest_dir
        packSize = self.opts.PackSize
        totFirings = self.opts.TotFirings
        numBoards = self.opts.NumBoards
        dataBlockSize = self.opts.DataBlockSize
        numElements = self.opts.NumElements
        # list files
        fileNameList = listdir(srcDir)
        chnDataList = [None] * (endInd - startInd + 1)
        for ind in range(startInd, endInd+1):
            print('Info: unpacking index %d...' % (ind))
            packData = []  # list of pack data
            # search through file list to find "experiment" (z step)
            # number for this particular index
            pattern = re.compile(r'Board([0-9]+)' +
                                 r'Experiment([0-9]+)TotalFiring' +
                                 str(totFirings) + '_Pack_' +
                                 str(ind) + '.bin')
            numExpr = -1
            for fileName in fileNameList:
                matchObj = pattern.match(fileName)
                if matchObj is not None:
                    _numExpr = int(matchObj.group(2))
                    if _numExpr != numExpr and numExpr != -1:
                        print('Warning: multiple' +
                              '\"experiment\" numbers found!' +
                              ' Last found will be used.')
                    numExpr = _numExpr
            if numExpr == -1:
                print('Warning: no file found. Skipping index %d' % (ind))
                chnDataList[ind-startInd] = None
                continue
            for boardId in range(numBoards):
                boardName = self.opts.BoardName[boardId]
                fileName = '%sExperiment%dTotalFiring%d_Pack_%d.bin' %\
                    (boardName, numExpr, totFirings, ind)
                filePath = join(srcDir, fileName)
                tempData =\
                    UnpackUtility.readBinFile(filePath, self.opts.dtype,
                                              packSize, totFirings, numExpr)
                if tempData is not None:
                    tempData = tempData[0:2*dataBlockSize, :]
                packData.append(tempData)
            if any([x is None for x in packData]):
                print('Warning: broken raw files. Skipping index %d' % (ind))
                chnDataList[ind-startInd] = None
                continue
            # interpret raw data into channel format
            # see daq_loop.c for original implementation
            chanMap = generateChanMap(numElements)
            chnData, chnDataAll = daq_loop(packData[0], packData[1],
                                           chanMap, numExpr)
            # fix bad channels
            chnData = -chnData/numExpr
            badChannels = [idx-1 for idx in self.opts.BadChannels]
            chnData[:, badChannels] = - chnData[:, badChannels]
            # save it to the list
            chnDataList[ind - startInd] = chnData
        timeSeqLenList = [UnpackUtility.sizeOfAxis(x, 0) for x in chnDataList]
        detectorNumList = [UnpackUtility.sizeOfAxis(x, 1) for x in chnDataList]
        timeSeqLen = max(timeSeqLenList)
        detectorNum = max(detectorNumList)
        numZStep = len(chnDataList)
        chnData3D = np.zeros((timeSeqLen, detectorNum, numZStep),
                             order='F', dtype=np.double)
        for idx in range(numZStep):
            chnData = chnDataList[idx]
            if chnData is not None:
                chnData3D[:, :, idx] = chnData
        chnData = np.mean(chnData3D, axis=2)
        # store indices in object
        self.startInd = startInd
        self.endInd = endInd
        # save data to object members
        self.chnData = chnData
        self.chnData3D = chnData3D

    def unpack(self):
        self.readChannelData()
        filename = 'scan_%d_%d.h5' % (self.startInd, self.endInd)
        outputPath = join(self.opts.dest_dir, filename)
        print('Saving file to %s ...' % (outputPath))
        f = h5py.File(outputPath, 'w')
        f['chndata'] = self.chnData
        f['chndata_all'] = self.chnData3D
        f.close()


class Unpack:

    def __init__(self, opts):
        assert(isinstance(opts, Options))
        self.opts = opts
        if not hasattr(self.opts, 'dest_ext'):
            self.opts.dest_ext = 'unpack'
        if not hasattr(self.opts, 'EXP_START'):
            self.opts.EXP_START = -1
            self.opts.EXP_END = -1
            self.opts.NUM_EXP = -1
        if not hasattr(self.opts, 'dtype'):
            self.opts.dtype = '<u4'
        # normalize and expand the paths
        self.opts.src_dir = normpath(expanduser(self.opts.src_dir))
        self.opts.dest_dir = normpath(expanduser(join(self.opts.src_dir,
                                                      self.opts.dest_ext)))

    def readChannelData(self):
        # variable extraction
        startInd = self.opts.EXP_START
        endInd = self.opts.EXP_END
        srcDir = self.opts.src_dir
        # destDir = self.opts.dest_dir
        packSize = self.opts.PackSize
        totFirings = self.opts.TotFirings
        numBoards = self.opts.NumBoards
        dataBlockSize = self.opts.DataBlockSize
        numElements = self.opts.NumElements
        # find next index
        if startInd == -1 or endInd == -1:
            nextInd = UnpackUtility.renameUnindexedFile(srcDir)
            if nextInd == -1:
                return
            startInd = nextInd
            endInd = nextInd
        fileNameList = listdir(srcDir)
        chnDataAllList = [None] * (endInd - startInd + 1)
        for ind in range(startInd, endInd+1):
            packData = []  # list of pack data
            # search through file list to find "experiment" (z step)
            # number for this particular index
            pattern = re.compile(r'Board([0-9]+)' +
                                 r'Experiment([0-9]+)TotalFiring' +
                                 str(totFirings) + '_Pack_' +
                                 str(ind) + '.bin')
            numExpr = -1
            for fileName in fileNameList:
                matchObj = pattern.match(fileName)
                if matchObj is not None:
                    _numExpr = int(matchObj.group(2))
                    if _numExpr != numExpr and numExpr != -1:
                        print('Warning: multiple' +
                              '\"experiment\" numbers found!' +
                              ' Last found will be used.')
                    numExpr = _numExpr
            if numExpr == -1:
                print('Warning: no file found. Skipping index %d' % (ind))
                continue  # no file to process. skip this index
            for boardId in range(numBoards):
                boardName = self.opts.BoardName[boardId]
                fileName = '%sExperiment%dTotalFiring%d_Pack_%d.bin' %\
                    (boardName, numExpr, totFirings, ind)
                filePath = join(srcDir, fileName)
                tempData =\
                    UnpackUtility.readBinFile(filePath, self.opts.dtype,
                                              packSize, totFirings, numExpr)
                if tempData is not None:
                    tempData = tempData[0:2*dataBlockSize, :]
                packData.append(tempData)
            if any([x is None for x in packData]):
                print('Warning: broken raw files. Skipping index %d' % (ind))
                continue
            # interpret raw data into channel format
            # see daq_loop.c for original implementation
            chanMap = generateChanMap(numElements)
            print('Starting daq_loop...')
            chnData, chnDataAll = daq_loop(packData[0], packData[1],
                                           chanMap, numExpr)
            # fix bad channels
            chnData = -chnData/numExpr
            badChannels = [idx-1 for idx in self.opts.BadChannels]
            chnData[:, badChannels] = - chnData[:, badChannels]
            chnDataAll = np.reshape(chnDataAll,
                                    (dataBlockSize, numElements, numExpr),
                                    order='F')
            chnDataAll[:, badChannels, :] = -chnDataAll[:, badChannels, :]

            chnDataAllList[ind - startInd] = chnDataAll
        zSteps = [UnpackUtility.sizeOfAxis(x, 2) for x in chnDataAllList]
        timeSeqLenList = [UnpackUtility.sizeOfAxis(x, 0)
                          for x in chnDataAllList]
        detectorNumList = [UnpackUtility.sizeOfAxis(x, 1)
                           for x in chnDataAllList]
        numZStep = sum(zSteps)
        timeSeqLen = max(timeSeqLenList)
        detectorNum = max(detectorNumList)
        chnData3D = np.zeros((timeSeqLen, detectorNum, numZStep),
                             order='F', dtype=np.double)
        zInd = 0
        for chnDataAll in chnDataAllList:
            if chnDataAll is not None:
                zSize = chnDataAll.shape[2]
                chnData3D[:, :, zInd:zInd+zSize] = chnDataAll
                zInd += zSize
        chnData = np.mean(chnData3D, axis=2)
        # store indices in object
        self.startInd = startInd
        self.endInd = endInd
        # save data to object members
        self.chnData = chnData
        self.chnData3D = chnData3D

    def unpack(self):
        self.readChannelData()
        UnpackUtility.saveChnData(self.chnData, self.chnData3D,
                                  self.opts.dest_dir, self.startInd)


class ReconUtility:

    @staticmethod
    def findDelayIdx(paData, fs):
        """
        find the delay value from the first few samples on A-lines
        """
        nSteps = paData.shape[1]
        refImpulse = paData[0:100, :]
        refImpulseEnv = np.abs(spsig.hilbert(refImpulse, axis=0))
        impuMax = np.amax(refImpulseEnv, axis=0)
        # to be consistent with the MATLAB's implementation ddof=1
        tempStd = np.std(refImpulseEnv, axis=0, ddof=1)
        delayIdx = -np.ones(nSteps)*18/fs
        for n in range(nSteps):
            if (impuMax[n] > 3.0*tempStd[n] and impuMax[n] > 0.1):
                tmpThresh = 2*tempStd[n]
                m1 = 14
                for ii in range(14, 50):
                    if refImpulse[ii-1, n] > -tmpThresh and\
                            refImpulse[ii, n] < -tmpThresh:
                        m1 = ii
                        break
                m2 = m1
                m3 = m1
                for ii in range(9, m1+1):
                    if refImpulse[ii-1, n] < tmpThresh and\
                            refImpulse[ii, n] > tmpThresh:
                        m2 = ii
                    if refImpulse[ii-1, n] > tmpThresh and\
                            refImpulse[ii, n] < tmpThresh:
                        m3 = ii
                delayIdx[n] = -float(m2+m3+2)/2.0/fs
        return delayIdx

    @staticmethod
    def updateProgress(current, total, timeRemain=None):
        """update progress bar"""
        TOTAL_INDICATOR_NUM = 50
        CHAR_INDICATOR = '#'
        progress = int(float(current)/total * 100)
        numIndicator = int(float(current)/total * TOTAL_INDICATOR_NUM)
        msg = '\r{:>3}% [{:<50}]'.format(progress, CHAR_INDICATOR*numIndicator)
        if timeRemain is not None:
            msg = msg + '\t {:.2f} mins remaining'.format(timeRemain)
        sys.stdout.write(msg)
        sys.stdout.flush()
        if current == total:
            print('\tDone')


class Reconstruction2D:

    def __init__(self, opts):
        assert(isinstance(opts, Options))
        self.opts = opts
        self.initialized = False

    def initRecon(self, paData):
        if self.initialized:
            return
        # parameters
        iniAngle = self.opts.iniAngle
        vm = self.opts.vm
        xSize = self.opts.xSize
        ySize = self.opts.ySize
        spacing = self.opts.spacing
        xCenter = self.opts.xCenter
        yCenter = self.opts.yCenter
        fs = self.opts.fs
        R = self.opts.R
        autoDelay = self.opts.autoDelay
        delay = self.opts.delay
        # start here
        (nSamples, nSteps, zSteps) = paData.shape
        anglePerStep = 2*np.pi/nSteps
        nPixelx = int(round(xSize / spacing))
        nPixely = int(round(ySize / spacing))
        # note range is 0-start indices
        xRange = (np.arange(1, nPixelx + 1, 1, dtype=np.double)
                  - nPixelx / 2) * xSize / nPixelx + xCenter
        yRange = (np.arange(nPixely, 0, -1, dtype=np.double)
                  - nPixely / 2) * ySize / nPixely + yCenter
        xImg = np.dot(np.ones((nPixely, 1)), xRange.reshape((1, nPixelx)))
        yImg = np.dot(yRange.reshape((nPixely, 1)), np.ones((1, nPixelx)))
        # xImg = np.copy(xImg, order='F')
        # yImg = np.copy(yImg, order='F')
        # receiver position
        detectorAngle = np.arange(0, nSteps, 1, dtype=np.double) *\
            anglePerStep + iniAngle/180.0*np.pi
        xReceive = np.cos(detectorAngle)*R
        yReceive = np.sin(detectorAngle)*R
        if autoDelay:
            # use the fisrt z step data to calibrate DAQ delay
            delayIdx = ReconUtility.findDelayIdx(paData[:, :, 0], fs)
        else:
            delayIdx = delay * np.ones(nSteps)
        # find index map and angular weighting for backprojection
        print('Calculating geometry dependent back-projection paramters')
        (self.idxAll, self.angularWeight, self.totalAngularWeight) =\
            find_index_map_and_angular_weight(nSteps, xImg, yImg, xReceive,
                                              yReceive, delayIdx, vm, fs)
        # reconstructed image buffer
        self.reImg = np.zeros((nPixely, nPixelx, zSteps), order='F')
        # store parameters
        self.nSamples = nSamples
        self.nSteps = nSteps
        self.zSteps = zSteps
        self.nPixelx = nPixelx
        self.nPixely = nPixely
        # set flag
        self.initialized = True

    def backprojection(self, paData):
        nSamples = self.nSamples
        # nSteps = self.nSteps
        zSteps = self.zSteps
        # nPixelx = self.nPixelx
        # nPixely = self.nPixely
        self.idxAll[self.idxAll > nSamples] = 1
        # back-projection
        print('Back-projection starts...')
        for z in range(zSteps):
            # remove DC
            paDataDC = np.dot(np.ones((nSamples - 99, 1)),
                              np.mean(paData[99:nSamples, :, z],
                                      axis=0).reshape((1, paData.shape[1])))
            paData[99:nSamples, :, z] = paData[99:nSamples, :, z] - paDataDC
        # speedup implementation
        self.reImg = backproject_loop(paData, self.idxAll, self.angularWeight,
                                      self.totalAngularWeight)
        """
        for z in range(zSteps):
            temp = np.copy(paData[:, :, z], order='F')
            paImg = recon_loop(temp, self.idxAll, self.angularWeight,
                               nPixelx, nPixely, nSteps)
            if paImg is None:
                print('WARNING: None returned as 2D reconstructed image!')
            paImg = paImg/self.totalAngularWeight
            self.reImg[:, :, z] = paImg
            ReconUtility.updateProgress(z+1, zSteps)
        """

    def reconstruct(self, paData):
        if paData.ndim == 2:
            (nSamples, nSteps) = paData.shape
            paData = np.reshape(paData, (nSamples, nSteps, 1))
        self.initRecon(paData)
        if self.opts.wiener:
            print('Wiener filtering raw data...')
            paData = subfunc_wiener(paData)
        if self.opts.exact:
            print('Filtering raw data for exact reconstruction...')
            paData = subfunc_exact(paData)
        self.backprojection(paData)
        # try to correct for mean value of the image
        self.reImg = self.reImg - np.mean(self.reImg.flatten())
        return self.reImg


class Reconstruction2DUnipolarHilbert(Reconstruction2D):

    def __init__(self, opts):
        super().__init__(opts)

    def reconstruct(self, paData):
        if paData.ndim == 2:
            (nSamples, nSteps) = paData.shape
            paData = np.reshape(paData, (nSamples, nSteps, 1))
        (nSamples, nSteps, zSteps) = paData.shape
        reImg1 = np.copy(super().reconstruct(paData))
        # take 90-degree phase shift
        import scipy.fftpack as spfp
        for z in range(zSteps):
            for n in range(nSteps):
                paData[:, n, z] = spfp.hilbert(paData[:, n, z])
        reImg2 = np.copy(super().reconstruct(paData))
        self.reImg = np.sqrt(reImg1 ** 2 + reImg2 ** 2)
        return self.reImg


class Reconstruction2DUnipolarMultiview(Reconstruction2D):

    def __init__(self, opts, sectorSize=64, sectorStep=64):
        Reconstruction2D.__init__(self, opts)
        self.sectorSize = sectorSize
        self.sectorStep = sectorStep

    def singleSector(self, paSlice, startInd):
        endInd = startInd + self.sectorSize
        indRange = list(range(startInd, endInd)) + \
            list(range(startInd+floor(self.nSteps/2),
                       endInd+floor(self.nSteps/2)))
        indRange = [ind if ind < 512 else ind-512 for ind in indRange]
        idxAll = np.copy(self.idxAll[:, :, indRange], order='F')
        angularWeight = np.copy(self.angularWeight[:, :, indRange],
                                order='F')
        temp = np.copy(paSlice[:, indRange], order='F')
        paImg = recon_loop(temp, idxAll, angularWeight,
                           self.nPixelx, self.nPixely, 2*self.sectorSize)
        angle = self.opts.iniAngle - 90.0 +\
            (startInd + self.sectorSize/2.0) * 360.0 / self.nSteps
        paImg = spnd.interpolation.rotate(paImg, -angle, reshape=False)
        paImg = np.abs(spsig.hilbert(paImg, axis=0))
        paImg = spnd.interpolation.rotate(paImg, angle, reshape=False)
        return paImg

    def doRecon(self, paData):
        nSamples = self.nSamples
        print('Reconstruction starts...')
        for z in range(self.zSteps):
            # remove DC
            paDataDC = np.dot(np.ones((nSamples - 99, 1)),
                              np.mean(paData[99:nSamples, :, z],
                                      axis=0).reshape((1, paData.shape[1])))
            paData[99:nSamples, :, z] = paData[99:nSamples, :, z] - paDataDC
            paSlice = np.copy(paData[:, :, z], order='F')
            reImgSlice = np.zeros((self.nPixely, self.nPixelx), order='F')
            for startInd in range(0, floor(self.nSteps/2), self.sectorStep):
                paImg = self.singleSector(paSlice, startInd)
                reImgSlice = reImgSlice + paImg
            self.reImg[:, :, z] = reImgSlice
            ReconUtility.updateProgress(z+1, self.zSteps)

    def reconstruct(self, paData):
        self.initRecon(paData)
        if self.opts.wiener:
            print('Wiener filtering raw data...')
            paData = subfunc_wiener(paData)
        if self.opts.exact:
            print('Filtering raw data for exact reconstruction...')
            paData = subfunc_exact(paData)
        self.doRecon(paData)
        return self.reImg


from io import StringIO
try:
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
except:
    NO_CUDA = True
from time import time
import os.path


class Reconstruction3D:

    SCRIPT_PATH = os.path.dirname(os.path.relpath(__file__))
    PULSE_STR = ""
    PULSE_FILE = os.path.join(SCRIPT_PATH, 'PULSE_ARRANGEMENTS.txt')
    with open(PULSE_FILE) as fid:
        PULSE_STR = fid.read()
    KERNEL_CU_FILE = os.path.join(SCRIPT_PATH, 'reconstruct_kernel.cu')

    def __init__(self, opts):
        assert(isinstance(opts, Options))
        self.opts = opts

    @staticmethod
    def rearrangePAData(paData):
        """rearrange paData array according to 8 firing order
        """
        nSamples, nSteps, zSteps = paData.shape
        assert nSteps == 512
        # load pulse list from the pre-defined string
        sio = StringIO(Reconstruction3D.PULSE_STR)
        pulseList = np.loadtxt(sio, dtype=np.int)
        # pulse list was created based on MATLAB's 1-starting index
        # minus 1 to correct for the 0-starting index in Python
        pulseList = pulseList - 1
        numGroup = pulseList.shape[0]
        paDataE = np.zeros((nSamples, nSteps / numGroup, zSteps * numGroup),
                           dtype=np.float32, order='F')
        for zi in range(zSteps):
            for fi in range(numGroup):
                paDataE[:, :, zi * numGroup + fi] =\
                    paData[:, pulseList[fi, :], zi]
        return paDataE, pulseList, numGroup

    def reconstruct(self, paData):
        # preprocess
        if self.opts.wiener:
            print('Wiener filtering raw data...')
            paData = subfunc_wiener(paData)
        if self.opts.exact:
            print('Filtering raw data for exact reconstruction...')
            paData = subfunc_exact(paData)
        # initialize CUDA
        cuda.init()
        dev = cuda.Device(0)
        ctx = dev.make_context()
        # parameters
        iniAngle = self.opts.iniAngle
        zPerStep = self.opts.zPerStep
        vm = self.opts.vm
        xSize = self.opts.xSize
        ySize = self.opts.ySize
        zSize = self.opts.zSize
        if zSize == 'full':
            zSize = self.opts.zPerStep * paData.shape[2]
        spacing = self.opts.spacing
        zSpacing = self.opts.zSpacing
        xCenter = self.opts.xCenter
        yCenter = self.opts.yCenter
        fs = self.opts.fs
        R = self.opts.R
        lenR = self.opts.lenR
        elementHeight = self.opts.elementHeight
        # calculate delay indices
        delayIdx = ReconUtility.findDelayIdx(paData[:, :, 0], fs)
        delayIdx = delayIdx.astype(np.float32)
        print('Re-arranging raw RF data according to firing squence')
        paData = paData.astype(np.float32)
        paData, pulseList, numGroup = self.rearrangePAData(paData)
        nSamples, nSteps, zSteps = paData.shape
        # notice the z step size is divided by firing group count
        zPerStep = zPerStep / numGroup
        zCenter = zPerStep * zSteps / 2
        # notice nSteps is now 64!!
        anglePerStep = 2 * np.pi / nSteps / numGroup
        nPixelx = int(round(xSize / spacing))
        nPixely = int(round(ySize / spacing))
        nPixelz = int(round(zSize / zSpacing))
        # note range is 0-start indices
        xRange = (np.arange(1, nPixelx + 1, 1, dtype=np.float32)
                  - nPixelx / 2) * xSize / nPixelx + xCenter
        yRange = (np.arange(nPixely, 0, -1, dtype=np.float32)
                  - nPixely / 2) * ySize / nPixely + yCenter
        zRange = (np.arange(1, nPixelz + 1, 1, dtype=np.float32)
                  - nPixelz / 2) * zSize / nPixelz + zCenter
        # receiver position
        angleStep1 = iniAngle / 180.0 * np.pi
        detectorAngle = np.arange(0, nSteps * numGroup, 1, dtype=np.float32)\
            * anglePerStep + angleStep1
        xReceive = np.cos(detectorAngle) * R
        yReceive = np.sin(detectorAngle) * R
        zReceive = np.arange(0, zSteps, dtype=np.float32) * zPerStep
        # create buffer on GPU for reconstructed image
        self.reImg = np.zeros((nPixely, nPixelx, nPixelz),
                              order='C', dtype=np.float32)
        d_reImg = cuda.mem_alloc(self.reImg.nbytes)
        cuda.memcpy_htod(d_reImg, self.reImg)
        d_cosAlpha = cuda.mem_alloc(nPixely * nPixelx * nSteps * numGroup * 4)
        d_tempc = cuda.mem_alloc(nPixely * nPixelx * nSteps * numGroup * 4)
        d_paDataLine = cuda.mem_alloc(nSamples * 4)
        # back projection loop
        print('Reconstruction starting. Keep patient.')
        d_xRange = cuda.mem_alloc(xRange.nbytes)
        cuda.memcpy_htod(d_xRange, xRange)
        d_yRange = cuda.mem_alloc(yRange.nbytes)
        cuda.memcpy_htod(d_yRange, yRange)
        d_zRange = cuda.mem_alloc(zRange.nbytes)
        cuda.memcpy_htod(d_zRange, zRange)
        d_xReceive = cuda.mem_alloc(xReceive.nbytes)
        cuda.memcpy_htod(d_xReceive, xReceive)
        d_yReceive = cuda.mem_alloc(yReceive.nbytes)
        cuda.memcpy_htod(d_yReceive, yReceive)
        # get module right before execution of function
        MOD = SourceModule(open(self.KERNEL_CU_FILE, 'r').read())
        precomp = MOD.get_function('calculate_cos_alpha_and_tempc')
        bpk = MOD.get_function('backprojection_kernel_fast')
        # compute cosAlpha and tempc
        st_all = time()
        precomp(d_cosAlpha, d_tempc, d_xRange, d_yRange,
                d_xReceive, d_yReceive, np.float32(lenR),
                grid=(nPixelx, nPixely), block=(nSteps * numGroup, 1, 1))
        ctx.synchronize()
        print('Done pre-computing cosAlpha and tempc.')
        st = time()
        for zi in range(zSteps):
            # find out the index of fire at each virtual plane
            fi = zi % numGroup
            for ni in range(nSteps):
                # transducer index
                ti = pulseList[fi, ni]
                cuda.memcpy_htod(d_paDataLine, paData[:, ni, zi])
                bpk(d_reImg, d_paDataLine, d_cosAlpha, d_tempc,
                    d_zRange, zReceive[zi], np.float32(lenR),
                    np.float32(elementHeight), np.float32(vm),
                    delayIdx[ti], np.float32(fs), np.uint32(ti),
                    np.uint32(nSteps * numGroup), np.uint32(nSamples),
                    grid=(nPixelx, nPixely), block=(nPixelz, 1, 1))
            et = time()
            time_remaining = ((zSteps - zi - 1) * (et - st) / (zi + 1)) / 60.0
            ReconUtility.updateProgress(zi + 1, zSteps, time_remaining)
        cuda.memcpy_dtoh(self.reImg, d_reImg)
        et_all = time()
        totalTime = (et_all - st_all) / 60.0
        print('Total time elapsed: {:.2f} mins'.format(totalTime))
        ctx.pop()
        del ctx
        return self.reImg
