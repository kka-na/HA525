
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

void gpu_ColorFilter(uchar *pcuSrc, uchar *pcuDst, int w, int h, int c)
{
	dim3 grid = dim3(w / 16, h / 16);										 // (w/16)x(h/16) thread blocks
	dim3 block = dim3(16, 16, 1);											 // 16x16x1 thread per block
	color_Filter2D<<<grid, block, sizeof(uchar)>>>(pcuSrc, w, h, c, pcuDst); // shared memory
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