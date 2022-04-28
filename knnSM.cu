#include<stdio.h>
#include<stdlib.h>
#include <opencv2/opencv.hpp>
#include <cfloat>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>

__device__ bool smaller(uchar3 a, uchar3 b) {
  float fa = float(a.x) + float(a.y) + float(a.z);
  float fb = float(b.x) + float(b.y) + float(b.z);
  return fa < fb;
}

__global__ void process(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> dst, int rows, int cols, int ksize, int percent )
{
  const int col = blockDim.x * blockIdx.x + threadIdx.x;
  const int row = blockDim.y * blockIdx.y + threadIdx.y;

  int sizeDiv2 = ksize/2;
  __shared__ uchar3 temp[32 + 15][32 + 15];
  int xindex = threadIdx.x + sizeDiv2;
  int yindex = threadIdx.y + sizeDiv2;
  if ((col > ksize && col < cols - 32) && (row > ksize && row < rows - 32)) {
    
    if (threadIdx.x < sizeDiv2 && threadIdx.y < sizeDiv2) {
      temp[yindex - sizeDiv2][xindex - sizeDiv2]  = src(row - sizeDiv2, col - sizeDiv2);
      temp[yindex + 32][xindex - sizeDiv2]  = src(row + 32, col - sizeDiv2);
      temp[yindex - sizeDiv2][xindex + 32] = src(row - sizeDiv2, col + 32);
      temp[yindex + 32][xindex + 32] = src(row + 32, col + 32);
    }

    if (threadIdx.x < sizeDiv2) {
      temp[yindex] [xindex - sizeDiv2] = src(row, col - sizeDiv2);
      temp[yindex] [xindex + 32] = src(row, col + 32);
    }

    if (threadIdx.y < sizeDiv2) {
      temp[yindex - sizeDiv2][xindex] = src(row- sizeDiv2, col );
      temp[yindex + 32][xindex] = src(row + 32, col);
    }
    

    //temp[yindex][xindex] = src(row, col);
  }
  __syncthreads();
  int startM_r = row - sizeDiv2;
  int startM_c = col - sizeDiv2;

  //uchar3 val = src(dst_y, dst_x);
  float total = 0.0;
  int val = (ksize-1)/2;
  int count = 0;
  uchar3 sorted[255];

  for (int i = 0; i < ksize; i++) {
    for (int j = 0; j < ksize; j++) {
      if ((startM_r + i) >= 0 && (startM_r + i) < rows) {
        if ((startM_c + j) >= 0 && (startM_c + j) < cols) {
          uchar3 key;
          if (threadIdx.x < sizeDiv2 && threadIdx.y < sizeDiv2
            && threadIdx.x > 32 - sizeDiv2 && threadIdx.y > 32 -sizeDiv2)
            key = temp[threadIdx.y + j][threadIdx.x + i]; 
          else key = src(startM_r + j, startM_c + i);
          int k = count - 1;
          while (k >= 0 && smaller(key, sorted[k])){
              sorted[k + 1] = sorted[k];
              --k;
          }
          sorted[k + 1] = key;
          count++;
        }
      }
    }
  }

  int hallf = int(((int((ksize*ksize) / 2) - 1) * percent) / 100);
  int ini = int((ksize*ksize) / 2) - hallf;
  int end = ((ksize*ksize) / 2) + hallf;
  if ((ksize*ksize) % 2 == 0) ++end;

  float b = 0.0;
  float g = 0.0;
  float r = 0.0;

  for (int k = ini; k <= end; k++) {
      r += float(sorted[k].x);
      g += float(sorted[k].y);
      b += float(sorted[k].z);
  }
  float div = float(end - ini + 1);

  unsigned char rf = r/div;
  unsigned char gf = g/div;
  unsigned char bf = b/div;

  dst(row, col).x = rf;
  dst(row, col).y = gf;
  dst(row, col).z = bf;
}

int divUp(int a, int b)
{
  return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void startCUDA (cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, int ksize, int percent)
{
  int THREADS = 32;
  int BLOCKS = (dst.cols + THREADS - 1) / THREADS;
  dim3 block_dim(THREADS, THREADS);
  dim3 grid_dim(BLOCKS, BLOCKS);

  process<<<grid_dim, block_dim>>>(src, dst, dst.rows, dst.cols, ksize, percent);

}

