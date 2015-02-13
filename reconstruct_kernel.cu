#include <math.h>

#define SIGN(x) ((x) > 0.0 ? 1 : -1)

__global__ void init_image_kernel(float *img) {
  size_t xi = blockIdx.x;
  size_t yi = blockIdx.y;
  size_t zi = threadIdx.x;
  size_t imgIdx = zi + yi*blockDim.x + xi*blockDim.x*gridDim.y;
  img[imgIdx] = 0.0;
}

__global__ void calculate_cos_alpha_and_tempc
(float *cosAlpha, float *tempc, float *xRange, float *yRange,
 float *xReceive, float *yReceive, float lenR) {
  size_t xi = blockIdx.x;
  size_t yi = blockIdx.y;
  size_t ni = threadIdx.x;
  size_t idx = ni + yi*blockDim.x + xi*blockDim.x*gridDim.y;
  float dx = xRange[xi] - xReceive[ni];
  float dy = yRange[yi] - yReceive[ni];
  float r0 = sqrt(xReceive[ni]*xReceive[ni] + yReceive[ni]*yReceive[ni]);
  float rr0 = sqrt(dx*dx + dy*dy);
  cosAlpha[idx] = fabs((-xReceive[ni]*dx-yReceive[ni]*dy)/r0/rr0);
  tempc[idx] = rr0 - lenR/cosAlpha[idx];
}

__global__ void backprojection_kernel_fast
(float *img, float *paDataLine,
 float *cosAlpha_, float *tempc_, float *zRange,
 float zReceive, float lenR, float elementHeight,
 float vm, float delayIdx, float fs,
 unsigned int ni, unsigned int nSteps, unsigned int lineLength) {
  size_t xi = blockIdx.x;
  size_t yi = blockIdx.y;
  size_t zi = threadIdx.x;
  size_t precompIdx = ni + yi*nSteps + xi*nSteps*gridDim.y;
  float dz = zRange[zi] - zReceive;
  float cosAlpha = cosAlpha_[precompIdx];
  float tempc = tempc_[precompIdx];
  if (fabs(dz/tempc) < fabs(elementHeight*cosAlpha/2.0/lenR)) {
    size_t imgIdx = zi + yi*blockDim.x + xi*blockDim.x*gridDim.y;
    float rr0 = sqrt(tempc*tempc + dz*dz)*SIGN(tempc) + lenR/cosAlpha;
    float angleWeightB = tempc/sqrt(tempc*tempc+dz*dz)*cosAlpha/(rr0*rr0);
    size_t idx0 = lround((rr0/vm-delayIdx)*fs);
    if (idx0 < lineLength) {
      img[imgIdx] += paDataLine[idx0] / angleWeightB;
    }
  }
}

__global__ void backprojection_2d_kernel_fast
(float *pa_img, float *pa_data, unsigned int *idxAll, float *angularWeight,
 int nSteps, int nTimeSamples) {
  size_t xi = blockIdx.x;
  size_t yi = blockIdx.y;
  size_t zi = threadIdx.x;
  size_t imgIdx = xi + yi*gridDim.x + zi*gridDim.x*gridDim.y;
  // all data arrays are in 'F' order
  for (size_t iStep = 0; iStep < nSteps; iStep++) {
    size_t idx = xi + yi*gridDim.x + iStep*gridDim.x*gridDim.y;
    pa_img[imgIdx] +=
      pa_data[(size_t)(idxAll[idx] - 1 + nTimeSamples*iStep + zi*nTimeSamples*nSteps)] * angularWeight[idx];
  }
}
