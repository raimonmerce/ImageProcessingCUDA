#include<stdio.h>
#include<stdlib.h>
#include <opencv2/opencv.hpp>
#include <cfloat>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>

__global__ void horizontal(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> dst, int rows, int cols, float *d_vec, int ksize)
{
  const int col = blockDim.x * blockIdx.x + threadIdx.x;
  const int row = blockDim.y * blockIdx.y + threadIdx.y;

  int sizeDiv2 = ksize/2;
  //MAX KSIZE = 15
  __shared__ uchar3 temp[32][32 + 15];

  int xindex = threadIdx.x + sizeDiv2;
  int yindex = threadIdx.y + sizeDiv2;
  if (col > ksize && col < cols - 32) {
    if (threadIdx.y < sizeDiv2) {
      temp[yindex][xindex - sizeDiv2] = src(row, col  - sizeDiv2);
      temp[yindex][xindex + 32] = src(row, col + 32);
    }
  }
  __syncthreads();

  int startM_r = row - sizeDiv2;
  int startM_c = col - sizeDiv2;

  float total = 0.0;
  float r = 0.0;
  float g = 0.0;
  float b = 0.0;

  for (int i = 0; i < ksize; i++) {
    if ((startM_r + i) >= 0 && (startM_r + i) < rows) {
      uchar3 val;
      if (threadIdx.x < sizeDiv2 && threadIdx.x > 32 - sizeDiv2){
        val = temp[threadIdx.y][threadIdx.x + i];   
      } else val = src(startM_r, startM_c + i);
      r += float(val.x) * d_vec[i];
      g += float(val.y) * d_vec[i];
      b += float(val.z) * d_vec[i];
      total += d_vec[i];
    }
  }
  unsigned char rf = r/total;
  unsigned char gf = g/total;
  unsigned char bf = b/total;

  dst(row, col).x = rf;
  dst(row, col).y = gf;
  dst(row, col).z = bf;
}

__global__ void vertical(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> dst, int rows, int cols, float *d_vec, int ksize)
{
  const int col = blockDim.x * blockIdx.x + threadIdx.x;
  const int row = blockDim.y * blockIdx.y + threadIdx.y;

  int sizeDiv2 = ksize/2;
  //MAX KSIZE = 15
  __shared__ uchar3 temp[32 + 15][32];

  int xindex = threadIdx.x + sizeDiv2;
  int yindex = threadIdx.y + sizeDiv2;
  if (row > ksize && row < rows - 32) {
    if (threadIdx.y < sizeDiv2) {
      temp[yindex - sizeDiv2][xindex] = src(row - sizeDiv2, col );
      temp[yindex + 32][xindex] = src(row + 32, col);
    }
  }
  __syncthreads();

  int startM_r = row - (ksize/2);
  int startM_c = col - (ksize/2);

  float total = 0.0;
  float r = 0.0;
  float g = 0.0;
  float b = 0.0;

  for (int j = 0; j < ksize; j++) {
    if ((startM_c + j) >= 0 && (startM_c + j) < cols) {
            uchar3 val;
      if (threadIdx.y < sizeDiv2 && threadIdx.y > 32 -sizeDiv2){
        val = temp[threadIdx.y + j][threadIdx.x];   
      } else val = src(startM_r + j, startM_c);
      r += float(val.x) * d_vec[j];
      g += float(val.y) * d_vec[j];
      b += float(val.z) * d_vec[j];
      total += d_vec[j];
    }
  }

  unsigned char rf = r/total;
  unsigned char gf = g/total;
  unsigned char bf = b/total;

  dst(row, col).x = rf;
  dst(row, col).y = gf;
  dst(row, col).z = bf;
}

int divUp(int a, int b)
{
  return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void startCUDA ( cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, cv::cuda::GpuMat& tmp, float *d_vec, int ksize)
//void startCUDA ( cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, float *d_vec, int ksize)
{
  int THREADS = 32;
  int BLOCKS = (dst.cols + THREADS - 1) / THREADS;
  dim3 block_dim(THREADS, THREADS);
  dim3 grid_dim(BLOCKS, BLOCKS);
  horizontal<<<grid_dim, block_dim>>>(src, tmp, dst.rows, dst.cols, d_vec, ksize);
  vertical<<<grid_dim, block_dim>>>(tmp, dst, dst.rows, dst.cols, d_vec, ksize);
}

