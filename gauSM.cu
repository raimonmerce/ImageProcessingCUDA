#include<stdio.h>
#include<stdlib.h>
#include <opencv2/opencv.hpp>
#include <cfloat>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>

__global__ void process(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> dst, int rows, int cols, float *d_matrix, int ksize)
{
  const int col = blockDim.x * blockIdx.x + threadIdx.x;
  const int row = blockDim.y * blockIdx.y + threadIdx.y;

  int sizeDiv2 = ksize/2;
  //MAX KSIZE = 15
  __shared__ uchar3 temp[32 + 15][32 + 15];

  //int gindex = threadIdx.x + blockIdx.x * blockDim.x;
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

  float total = 0.0;
  float r = 0.0;
  float g = 0.0;
  float b = 0.0;

  for (int i = 0; i < ksize; i++) {
    // Go over each column
    for (int j = 0; j < ksize; j++) {
      // Range check for rows
      if ((threadIdx.x + i) >= 0 && (threadIdx.x + i) < 47) {
        // Range check for columns
        if ((threadIdx.y + j) >= 0 && (threadIdx.y + j) < 47) {
          uchar3 val;
          if (threadIdx.x < sizeDiv2 && threadIdx.y < sizeDiv2
            && threadIdx.x > 32 - sizeDiv2 && threadIdx.y > 32 -sizeDiv2){
            val = temp[threadIdx.y + j][threadIdx.x + i];   
          } else val = src(startM_r + (j), startM_c + (i));
          r += float(val.x) * d_matrix[i * ksize + j];
          g += float(val.y) * d_matrix[i * ksize + j];
          b += float(val.z) * d_matrix[i * ksize + j];
          total += d_matrix[i * ksize + j];
        }
      }
    }
  }

  //printf("Hello from block %d, thread %f\n", blockIdx.x, total);

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

void startCUDA (cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, float *d_matrix, int ksize)
{
  int THREADS = 32;
  int BLOCKS = (dst.cols + THREADS - 1) / THREADS;
  dim3 block_dim(THREADS, THREADS);
  dim3 grid_dim(BLOCKS, BLOCKS);

  process<<<grid_dim, block_dim>>>(src, dst, dst.rows, dst.cols, d_matrix, ksize);

}

