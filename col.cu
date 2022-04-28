#include<stdio.h>
#include<stdlib.h>
#include <opencv2/opencv.hpp>
#include <cfloat>
#include <opencv2/core/cuda/common.hpp>
#include <opencv2/core/cuda/border_interpolate.hpp>
#include <opencv2/core/cuda/vec_traits.hpp>
#include <opencv2/core/cuda/vec_math.hpp>

__global__ void process(const cv::cuda::PtrStep<uchar3> src, cv::cuda::PtrStep<uchar3> dst, int rows, int cols, float rad )
{
 
  const int dst_x = blockDim.x * blockIdx.x + threadIdx.x;
  const int dst_y = blockDim.y * blockIdx.y + threadIdx.y;

  uchar3 full = make_uchar3 ( 255, 255, 255 );
  /* full.x=255; */
  /* full.y=255; */
  /* full.z=255; */
  
  if (dst_x < cols && dst_y < rows)
    {
      uchar3 val = src(dst_y, dst_x);

      float var_R = float(val.x) / 255.0;
      float var_G = float(val.y) / 255.0;
      float var_B = float(val.z) / 255.0;

      if (var_R > 0.04045) var_R = powf(((var_R + 0.055) / 1.055), 2.4);
      else var_R = var_R / 12.92;
      if (var_G > 0.04045) var_G = powf(((var_G + 0.055) / 1.055), 2.4);
      else var_G = var_G / 12.92;
      if (var_B > 0.04045) var_B = powf(((var_B + 0.055) / 1.055), 2.4);
      else var_B = var_B / 12.92;

      var_R *= 100.0;
      var_G *= 100.0;
      var_B *= 100.0;

      float XYZ[3];

      XYZ[0] = (var_R * 0.4124 + var_G * 0.3576 + var_B * 0.1805) / 95.047;
      XYZ[1] = (var_R * 0.2126 + var_G * 0.7152 + var_B * 0.0722) / 100.0;
      XYZ[2] = (var_R * 0.0193 + var_G * 0.1192 + var_B * 0.9505) / 108.883;

      for (int i = 0; i < 3; ++i) {
          float value = XYZ[i];
          if (value > 0.008856) {
              value = powf(value, 0.3333333333333333);
          } else {
              value = (7.787 * value) + float(16 / 116);
          }
          XYZ[i] = value;
      }

      float L = (116.0 * XYZ[1]) - 16.0;
      float a = 500.0 * (XYZ[0] - XYZ[1]);
      float b = 200.0 * (XYZ[1] - XYZ[2]);

      float C = sqrtf((a * a) + (b * b));
      float hub = atan(b / a);

      float na = cos(hub + rad) * C;
      float nb = sin(hub + rad) * C;

      float nY = (L + 16.0) / 116.0;
      float nX = (na / 500.0) + nY;
      float nZ = nY - (nb / 200.0);

      if (powf(nY, 3.0) > 0.008856) nY = powf(nY, 3.0);
      else                       nY = (nY - (16.0 / 116.0)) / 7.787;
      if (powf(nX, 3.0) > 0.008856) nX = powf(nX, 3.0);
      else                       nX = (nX - (16.0 / 116.0)) / 7.787;
      if (powf(nZ, 3.0) > 0.008856) nZ = powf(nZ, 3.0);
      else                       nZ = (nZ - (16.0 / 116.0)) / 7.787;

      nX *= 95.047;
      nY *= 100.0;
      nZ *= 108.883;

      float nR = (nX / 100.0) * 3.2406 + (nY / 100.0) * -1.5372 + (nZ / 100.0) * -0.4986;
      float nG = (nX / 100.0) * -0.9689 + (nY / 100.0) * 1.8758 + (nZ / 100.0) * 0.0415;
      float nB = (nX / 100.0) * 0.0557 + (nY / 100.0) * -0.2040 + (nZ / 100.0) * 1.0570;

      if (nR > 0.0031308) nR = 1.055 * (powf(nR, (1.0 / 2.4))) - 0.055;
      else                nR = 12.92 * nR;
      if (nG > 0.0031308) nG = 1.055 * (powf(nG, (1.0 / 2.4))) - 0.055;
      else                nG = 12.92 * nG;
      if (nB > 0.0031308) nB = 1.055 * (powf(nB, (1.0 / 2.4))) - 0.055;
      else                nB = 12.92 * nB;

      //dst(dst_y, dst_x).x = uchar(nB * 255);
      //dst(dst_y, dst_x).y = uchar(nG * 255);
      //dst(dst_y, dst_x).z = uchar(nR * 255);
      unsigned char rf = nR * 255.0;
      unsigned char gf = nG * 255.0;
      unsigned char bf = nB * 255.0;
      //printf("Hello from block %d, thread %u\n", blockIdx.x, rf);
      dst(dst_y, dst_x).x = rf;
      dst(dst_y, dst_x).y = gf;
      dst(dst_y, dst_x).z = bf;
    }
}

int divUp(int a, int b)
{
  return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void startCUDA ( cv::cuda::GpuMat& src, cv::cuda::GpuMat& dst, float rad )
{
  const dim3 block(32, 8);
  const dim3 grid(divUp(dst.cols, block.x), divUp(dst.rows, block.y));

  process<<<grid, block>>>(src, dst, dst.rows, dst.cols, rad);

}

