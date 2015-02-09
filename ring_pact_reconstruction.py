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

# from unpack_speedup import daq_loop, generateChanMap
# from recon_loop import recon_loop, find_index_map_and_angular_weight
from ring_pact_speedup import daq_loop, generateChanMap
from ring_pact_speedup import recon_loop, find_index_map_and_angular_weight
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
    def updateProgress(current, total):
        """update progress bar"""
        TOTAL_INDICATOR_NUM = 50
        CHAR_INDICATOR = '#'
        progress = int(float(current)/total * 100)
        numIndicator = int(float(current)/total * TOTAL_INDICATOR_NUM)
        msg = '\r{:>3}% [{:<50}]'.format(progress, CHAR_INDICATOR*numIndicator)
        sys.stdout.write(msg)
        sys.stdout.flush()
        if current == total:
            print('\tDone')


class Reconstruction2D:

    def __init__(self, opts):
        assert(isinstance(opts, Options))
        self.opts = opts

    def initRecon(self, paData):
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
        xImg = np.copy(xImg, order='F')
        yImg = np.copy(yImg, order='F')
        # receiver position
        detectorAngle = np.arange(0, nSteps, 1, dtype=np.double) *\
            anglePerStep + iniAngle/180.0*np.pi
        xReceive = np.cos(detectorAngle)*R
        yReceive = np.sin(detectorAngle)*R
        # use the fisrt z step data to calibrate DAQ delay
        delayIdx = ReconUtility.findDelayIdx(paData[:, :, 0], fs)
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

    def backprojection(self, paData):
        nSamples = self.nSamples
        nSteps = self.nSteps
        zSteps = self.zSteps
        nPixelx = self.nPixelx
        nPixely = self.nPixely
        self.idxAll[self.idxAll > nSamples] = 1
        # back-projection
        print('Back-projection starts...')
        for z in range(zSteps):
            # remove DC
            paDataDC = np.dot(np.ones((nSamples - 99, 1)),
                              np.mean(paData[99:nSamples, :, z],
                                      axis=0).reshape((1, paData.shape[1])))
            paData[99:nSamples, :, z] = paData[99:nSamples, :, z] - paDataDC
            temp = np.copy(paData[:, :, z], order='F')
            paImg = recon_loop(temp, self.idxAll, self.angularWeight,
                               nPixelx, nPixely, nSteps)
            if paImg is None:
                print('WARNING: None returned as 2D reconstructed image!')
            paImg = paImg/self.totalAngularWeight
            self.reImg[:, :, z] = paImg
            ReconUtility.updateProgress(z+1, zSteps)
        # return the reconstructed image
        # return self.reImg

    def reconstruct(self, paData):
        self.initRecon(paData)
        if self.opts.wiener:
            print('Wiener filtering raw data...')
            paData = subfunc_wiener(paData)
        if self.opts.exact:
            print('Filtering raw data for exact reconstruction...')
            paData = subfunc_exact(paData)
        self.backprojection(paData)
        return self.reImg


class Reconstruction2DUnipolar(Reconstruction2D):

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


class Reconstruction3D:
    pass
