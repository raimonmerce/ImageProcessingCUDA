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

  int startM_r = row - (ksize/2);
  int startM_c = col - (ksize/2);

  float total = 0.0;
  float r = 0.0;
  float g = 0.0;
  float b = 0.0;

  for (int i = 0; i < ksize; i++) {
    // Go over each column
    for (int j = 0; j < ksize; j++) {
      // Range check for rows
      if ((startM_r + i) >= 0 && (startM_r + i) < rows) {
        // Range check for columns
        if ((startM_c + j) >= 0 && (startM_c + j) < cols) {
          uchar3 val = src(startM_r + (j), startM_c + (i));
          float tmp = d_matrix[i * ksize + j];
          r += float(val.x) * tmp;
          g += float(val.y) * tmp;
          b += float(val.z) * tmp;
          total += tmp;
        }
      }
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

void startCUDA (cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, float *d_matrix, int ksize)
{
  int THREADS = 32;
  int BLOCKS = (dst.cols + THREADS - 1) / THREADS;
  dim3 block_dim(THREADS, THREADS);
  dim3 grid_dim(BLOCKS, BLOCKS);

  process<<<grid_dim, block_dim>>>(src, dst, dst.rows, dst.cols, d_matrix, ksize);

}

