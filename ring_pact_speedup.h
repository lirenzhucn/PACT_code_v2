#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <omp.h>
//#include <sys/time.h>
//#include <unistd.h>

double round(double d) {
  return (d>0.0 ? floor(d+0.5) : floor(d-0.5));
}

// implementation function
void recon_loop_imp(const double *pa_data, const double *idxAll,
    const double *angularWeight, int nPixelx, int nPixely,
    int nSteps, int nTimeSamples, double *pa_img) {
  int iStep, y, x, icount, pcount, iskip;
  double paImgPixel;
  //struct timeval start, end, dif;
  //int iskips[nSteps];
  int iskips[512];

  //gettimeofday(&start, NULL);
  for (iStep=0; iStep<nSteps; iStep++) {
    iskips[iStep] = nTimeSamples * iStep - 1;
  }
  pcount = 0;
  icount = 0;
  for (x=0; x<nPixelx; x++) {
    for (y=0; y<nPixely; y++) {
      paImgPixel = 0.0;
      for (iStep=0; iStep<nSteps; iStep++) {
        iskip = iskips[iStep];
        double x = idxAll[icount];
        double x1 = floor(x);
        double x2 = ceil(x);
        double y;
        if (x1 != x2) {
          double y1 = pa_data[(int)x1+iskip];
          double y2 = pa_data[(int)x2+iskip];
          y = (y2-y1)*(x-x1) + y1;
        } else {
          y = pa_data[(int)x+iskip];
        }
        paImgPixel += y * angularWeight[icount];
	//paImgPixel +=
	  //pa_data[(int)round(idxAll[icount])+iskip] *
	  //angularWeight[icount];
        icount++;
      }
      pa_img[pcount]= paImgPixel;
      pcount++;
    }
  }
  //gettimeofday(&end, NULL);
  //timersub(&end, &start, &dif);
  //printf("Thread time %ld.%06ld s.\n",
      //(long int)dif.tv_sec, (long int)dif.tv_usec);
}

void backproject_loop_imp(const double *paData, const double *idxAll,
    const double *angularWeight, const double *totalAngularWeight,
    int nPixelx, int nPixely, int zSteps, int nSteps, int nSamples,
    double *paImg) {
  int z, ind;
#pragma omp parallel for
  for (z = 0; z < zSteps; z++) {
    const double *paDataPointer = paData + z*nSamples*nSteps;
    double *paImgPointer = paImg + z*nPixely*nPixelx;
    recon_loop_imp(paDataPointer, idxAll, angularWeight,
        nPixelx, nPixely, nSteps, nSamples, paImgPointer);
    for (ind = 0; ind < nPixely*nPixelx; ind++) {
      paImgPointer[ind] /= totalAngularWeight[ind];
    }
  }
}

void find_index_map_and_angular_weight_imp
(const int nSteps, const double *xImg, const double *yImg,
 const double *xReceive, const double *yReceive, const double *delayIdx,
 const double vm, const double fs, const long nSize2D,
 double *idxAll, double *angularWeight, double *totalAngularWeight) {
  int n, i;
  double r0, rr0, dx, dy, cosAlpha;
#pragma omp parallel for private(r0, rr0, dx, dy, cosAlpha)
  for (n=0; n<nSteps; n++) {
    r0 = sqrt(xReceive[n]*xReceive[n] + yReceive[n]*yReceive[n]);
    for (i=0; i<nSize2D; i++) {
      dx = xImg[i] - xReceive[n];
      dy = yImg[i] - yReceive[n];
      rr0 = sqrt(dx*dx + dy*dy);
      cosAlpha = fabs((-xReceive[n]*dx-yReceive[n]*dy)/r0/rr0);
      cosAlpha = cosAlpha<0.999 ? cosAlpha : 0.999;
      angularWeight[n + i*nSteps] = cosAlpha / (rr0 * rr0);
      totalAngularWeight[i] += angularWeight[n + i*nSteps];
      //idxAll[n + i*nSteps] = (uint64_t)round((rr0/vm - delayIdx[n]) * fs);
      idxAll[n + i*nSteps] = (rr0/vm - delayIdx[n]) * fs;
    }
  }
}

void generateChanMap_imp(const int numElements, uint32_t *chanMap) {
  uint32_t temp[128] = {1, 2, 3, 4, 68, 67, 66, 65,};
  uint32_t chanOffset[] = {1*128, 3*128, 2*128, 0*128};
  int n, i;
  for (n = 1; n < 16; n++) {
    for (i = 0; i < 8; i++) {
      temp[n*8 + i] = temp[(n-1)*8 + i] + 4;
    }
  }
  for (n = 1; n < 5; n++) {
    for (i = 0; i < 128; i++) {
      chanMap[(n-1)*128 + i] = temp[i] + chanOffset[n-1];
    }
  }
}

#define hex3ff 1023
#define DataBlockSize 1300
#define NumElements 512
#define TotFirings 8
#define NumDaqChnsBoard 32

void daq_loop_imp(uint32_t *packData1, uint32_t *packData2,
    uint32_t *chanMap, const int numExperiments, const int packSize,
    double *chndata, double *chndata_all) {
  double raw_data[TotFirings][NumDaqChnsBoard][DataBlockSize];
  int B, N, F, counter, i, j, channel, chan_offset;
  int chanindex, firingindex;
  uint32_t *pack_data;

  double mean_data;
  uint32_t pd0, pd1;

  for (B = 0; B < 2; B++) {
    counter = 0;
    chan_offset = B * TotFirings * NumDaqChnsBoard;
    if (B == 0) { pack_data = packData1; }
    else { pack_data = packData2; }
    for (N = 0; N < numExperiments; N++) {
      for (F = 0; F < TotFirings; F++) {
        i = 0; channel = 0;
        while (i < 6) {
          for (j = 0; j < DataBlockSize; j++) {
            // C-order arrays
            pd0 = pack_data[2*j*packSize + counter];
            pd1 = pack_data[(2*j+1)*packSize + counter];
            raw_data[F][channel][j] = (double)(pd0 & hex3ff);
            raw_data[F][channel+1][j] = (double)(pd1 & hex3ff);
            raw_data[F][channel+2][j] = (double)((pd0>>10) & hex3ff);
            raw_data[F][channel+3][j] = (double)((pd1>>10) & hex3ff);
            if (i!=2 && i!=5) {
              raw_data[F][channel+4][j] = (double)((pd0>>20) & hex3ff);
              raw_data[F][channel+5][j] = (double)((pd1>>20) & hex3ff);
            }
          }
          if (i!=2 && i!=5) {
            channel += 6;
          } else {
            channel += 4;
          }
          i ++; counter ++;
        }
      }
      for (chanindex = 0; chanindex < NumDaqChnsBoard; chanindex ++) {
        for (firingindex = 0; firingindex < TotFirings; firingindex ++) {
          mean_data = 0.0;
          for (i = 0; i < DataBlockSize; i++) {
            mean_data += raw_data[firingindex][chanindex][i];
          }
          mean_data /= DataBlockSize;
          for (i = 0; i < DataBlockSize; i++) {
            raw_data[firingindex][chanindex][i] =
              (raw_data[firingindex][chanindex][i] - mean_data)/NumElements;
          }
          channel = chanMap[chanindex*8+firingindex+chan_offset] - 1;
          for (i = 0; i < DataBlockSize; i++) {
            chndata[i+channel*DataBlockSize] +=
              raw_data[firingindex][chanindex][i];
            chndata_all[i+(N*NumElements+channel)*DataBlockSize] =
              -raw_data[firingindex][chanindex][i];
          }
        }
      }
    }
  }
}
