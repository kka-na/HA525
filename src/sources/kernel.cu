
#include </usr/local/cuda-11.4/include/cuda_runtime.h>
#include </usr/local/cuda-11.4/include/device_launch_parameters.h>

#include <opencv2/opencv.hpp>
#include <stdio.h>

using namespace cv;
using namespace std;

__global__ void color_Filter2D(uchar *pSrcImage, int SrcWidth, int SrcHeight, int SrcChannel, uchar *pDstImage)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int srcIndex = (y * SrcWidth + x) * SrcChannel;
	int dstIndex = y * SrcWidth + x;

	if (15 <= pSrcImage[srcIndex] && pSrcImage[srcIndex] <= 23 && 40 <= pSrcImage[srcIndex + 1] && pSrcImage[srcIndex + 1] <= 255 && 100 <= pSrcImage[srcIndex + 2] && pSrcImage[srcIndex + 2] <= 255)
	{
		pDstImage[dstIndex] = 255;
	}
	else if (0 <= pSrcImage[srcIndex] && pSrcImage[srcIndex] <= 255 && 0 <= pSrcImage[srcIndex + 1] && pSrcImage[srcIndex + 1] <= 35 && 200 <= pSrcImage[srcIndex + 2] && pSrcImage[srcIndex + 2] <= 255)
	{
		pDstImage[dstIndex] = 255;
	}
	else
	{
		pDstImage[dstIndex] = 0;
	}
}

__global__ void add_Frame2D(uchar *pSrcImage1, uchar *pSrcImage2, int SrcWidth, int SrcHeight, uchar *pDstImage)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int Index = y * SrcWidth + x;

	pDstImage[Index] = pSrcImage1[Index] + pSrcImage2[Index];
}

__global__ void mask_Lane2D(uchar *pSrcImage, uchar *pLaneImage, int SrcWidth, int SrcHeight, int SrcChannel)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int colorIndex = (y * SrcWidth + x) * SrcChannel;

	if (pLaneImage[colorIndex] + pLaneImage[colorIndex + 1] + pLaneImage[colorIndex + 2] != 0)
	{
		for (int i = 0; i < SrcChannel; i++)
		{
			pSrcImage[colorIndex + i] = int(pLaneImage[colorIndex + i]);
		}
	}
}

__global__ void perspectiveTransform2D(uchar *pTopViewImage, double *perspMatrix, int SrcWidth, int SrcHeight, int SrcChannel, uchar *pBEVImage)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int colorIndex = (y * SrcWidth + x) * SrcChannel;
	double new_x = (perspMatrix[0] * x + perspMatrix[1] * y + perspMatrix[2]) / (perspMatrix[6] * x + perspMatrix[7] * y + perspMatrix[8]);
	double new_y = (perspMatrix[3] * x + perspMatrix[4] * y + perspMatrix[5]) / (perspMatrix[6] * x + perspMatrix[7] * y + perspMatrix[8]);
	int new_colorIndex = ((int)new_y * SrcWidth + (int)new_x) * SrcChannel;

	if (0 <= new_x && new_x < SrcWidth && 0 <= new_y && new_y < SrcHeight)
	{
		for (int i = 0; i < SrcChannel; i++)
		{
			pBEVImage[new_colorIndex + i] = pTopViewImage[colorIndex + i];
		}
	}
}

__global__ void bilinearInterpolation2D(uchar *pBEVImage, double *perspMatrix, int SrcWidth, int SrcHeight, int SrcChannel, uchar *pTopViewImage)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int colorIndex = (y * SrcWidth + x) * SrcChannel;
	double new_x = (perspMatrix[0] * x + perspMatrix[1] * y + perspMatrix[2]) / (perspMatrix[6] * x + perspMatrix[7] * y + perspMatrix[8]);
	double new_y = (perspMatrix[3] * x + perspMatrix[4] * y + perspMatrix[5]) / (perspMatrix[6] * x + perspMatrix[7] * y + perspMatrix[8]);
	int new_colorIndex = ((int)new_y * SrcWidth + (int)new_x) * SrcChannel;

	if (0 <= new_x && new_x < SrcWidth && 0 <= new_y && new_y < SrcHeight)
	{
		if (pBEVImage[colorIndex] + pBEVImage[colorIndex + 1] + pBEVImage[colorIndex + 2] == 0)
		{
			double a = new_x - (int)new_x;
			double b = new_y - (int)new_y;
			for (int i = 0; i < SrcChannel; i++)
			{
				pBEVImage[colorIndex + i] = (1 - a) * (1 - b) * pTopViewImage[((int)new_y * SrcWidth + (int)new_x) * SrcChannel + i] + a * (1 - b) * pTopViewImage[((int)new_y * SrcWidth + (int)new_x + 1) * SrcChannel + i] + a * b * pTopViewImage[(((int)new_y + 1) * SrcWidth + (int)new_x + 1) * SrcChannel + i] + (1 - a) * b * pTopViewImage[((int)new_y * SrcWidth + (int)new_x + 1) * SrcChannel + i];
			}
		}
	}
}

__global__ void zero_Padding2D(uchar *pSrcImage, int SrcWidth, int SrcHeight, uchar *zPadImage)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int Index = y * SrcWidth + x;
	int paddedIndex = (y + 1) * (SrcWidth + 2) + x + 1;

	zPadImage[paddedIndex] = pSrcImage[Index];
}

__global__ void scharr_Filter2D(uchar *zPadImage, int SrcWidth, int SrcHeight, uchar *pDstImage)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int Index = y * SrcWidth + x;
	int paddedIndex = (y + 1) * (SrcWidth + 2) + x + 1;

	float filter_x[] = {3, 10, 3, 0, 0, 0, -3, -10, -3};
	float filter_y[] = {3, 0, -3, 10, 0, -10, 3, 0, -3};

	int sum = 0;
	sum += zPadImage[paddedIndex - SrcWidth - 3] * filter_x[0] + zPadImage[paddedIndex - SrcWidth - 2] * filter_x[1] + zPadImage[paddedIndex - SrcWidth - 1] * filter_x[2] + zPadImage[paddedIndex - 1] * filter_x[3] + zPadImage[paddedIndex] * filter_x[4] + zPadImage[paddedIndex + 1] * filter_x[5] + zPadImage[paddedIndex + SrcWidth + 1] * filter_x[6] + zPadImage[paddedIndex + SrcWidth + 2] * filter_x[7] + zPadImage[paddedIndex + SrcWidth + 3] * filter_x[8] + zPadImage[paddedIndex - SrcWidth - 3] * filter_y[0] + zPadImage[paddedIndex - SrcWidth - 2] * filter_y[1] + zPadImage[paddedIndex - SrcWidth - 1] * filter_y[2] + zPadImage[paddedIndex - 1] * filter_y[3] + zPadImage[paddedIndex] * filter_y[4] + zPadImage[paddedIndex + 1] * filter_y[5] + zPadImage[paddedIndex + SrcWidth + 1] * filter_y[6] + zPadImage[paddedIndex + SrcWidth + 2] * filter_y[7] + zPadImage[paddedIndex + SrcWidth + 3] * filter_y[8];
	if (sum < 0)
		sum = (-1) * sum;
	if (sum > 255)
		sum = 255;
	if (sum < 150)
		sum = 0;
	else
		sum = 255;

	pDstImage[Index] = (uchar)sum;
}

void gpu_ColorFilter(uchar *pcuSrc, uchar *pcuDst, int w, int h, int c)
{
	dim3 grid = dim3(w / 16, h / 16);										 // (w/16)x(h/16) thread blocks
	dim3 block = dim3(16, 16, 1);											 // 16x16x1 thread per block
	color_Filter2D<<<grid, block, sizeof(uchar)>>>(pcuSrc, w, h, c, pcuDst); // shared memory
	cudaThreadSynchronize();
}

void gpu_ScharrFilter(uchar *pcuSrc, uchar *zPadSrc, uchar *pcuDst, int w, int h)
{
	dim3 grid = dim3(w / 16, h / 16); // (w/16)x(h/16) thread blocks
	dim3 block = dim3(16, 16, 1);	  // 16x16x1 thread per block
	zero_Padding2D<<<grid, block, sizeof(uchar)>>>(pcuSrc, w, h, zPadSrc);
	cudaThreadSynchronize();
	scharr_Filter2D<<<grid, block, sizeof(uchar)>>>(zPadSrc, w, h, pcuDst);
	cudaThreadSynchronize();
}

void gpu_AddFrame(uchar *pcuSrc1, uchar *pcuSrc2, uchar *pcuDst, int w, int h)
{
	dim3 grid = dim3(w / 16, h / 16); // (w/16)x(h/16) thread blocks
	dim3 block = dim3(16, 16, 1);	  // 16x16x1 thread per block
	add_Frame2D<<<grid, block, sizeof(uchar)>>>(pcuSrc1, pcuSrc2, w, h, pcuDst);
	cudaThreadSynchronize();
}

void gpu_MaskingLane(uchar *pcuSrc1, uchar *pcuSrc2, int w, int h, int c)
{
	dim3 grid = dim3(w / 16, h / 16); // (w/16)x(h/16) thread blocks
	dim3 block = dim3(16, 16, 1);	  // 16x16x1 thread per block
	mask_Lane2D<<<grid, block, sizeof(uchar)>>>(pcuSrc1, pcuSrc2, w, h, c);
	cudaThreadSynchronize();
}

void gpu_PerspectiveTransform(uchar *pcuSrc, uchar *pcuDst, double *perspMatrix, double *perspMatrixInv, int w, int h, int c)
{
	dim3 grid = dim3(w / 16, h / 16); // (w/16)x(h/16) thread blocks
	dim3 block = dim3(16, 16, 1);	  // 16x16x1 thread per block
	perspectiveTransform2D<<<grid, block, sizeof(uchar)>>>(pcuSrc, perspMatrix, w, h, c, pcuDst);
	cudaThreadSynchronize();
	bilinearInterpolation2D<<<grid, block, sizeof(uchar)>>>(pcuDst, perspMatrixInv, w, h, c, pcuSrc);
	cudaThreadSynchronize();
}