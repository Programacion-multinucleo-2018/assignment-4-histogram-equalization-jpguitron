#include <iostream>
#include <cstdio>
#include <cmath>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "common.h"
#include <cuda_runtime.h>

#include <chrono>
//nvcc -o exe img_eq.cu -lopencv_core -lopencv_highgui -lopencv_imgproc -std=c++11
//se requieren minimo 256 threadspor bloque
//falta hacer por si se acaban los bloques


using namespace std;

__global__ void save_to_image(unsigned char* input, unsigned char* output, int width, int height, int step, int *temp)
{
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	const int tid = yIndex*step+xIndex;

	if((xIndex < width) && (yIndex < height))
	{
		int actual = input[tid];
		output[tid] = temp[actual];
	}
}

__global__ void equalization_phase2(unsigned char* input, unsigned char* output, float imgSize, int *temp)
{

	int yxn = threadIdx.x+threadIdx.y*blockDim.x;

	if(yxn < 256 && blockIdx.x == 0 && blockIdx.y==0)
	{
		temp[yxn] = temp[yxn]*(255/imgSize); 
	}
}

__global__ void equalization_phase1(unsigned char* input, unsigned char* output, int *temp)
{

	int yxn = threadIdx.x+threadIdx.y*blockDim.x;

	if(yxn < 256 && blockIdx.x ==0 && blockIdx.y==0)
	{
		int a = 0;
		for(int x = 0; x <= yxn; x++)
			a += temp[x];
		
		temp[yxn] = a;
	}

}

__global__ void generate_histogram(unsigned char* input, unsigned char* output, int width, int height, int step, int *temp)
{

	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	const int tid = yIndex*step+xIndex;

	__shared__ int h[256];

	int yxn = threadIdx.x+threadIdx.y*blockDim.x;

	if(yxn < 256)
			h[yxn] = 0;

	__syncthreads();
	
	if((xIndex < width) && (yIndex < height))
	{
		atomicAdd(&h[input[tid]], 1);
	}

	__syncthreads();

	if(yxn < 256)
		atomicAdd(&temp[yxn], h[yxn]);

}

void equalization(const cv::Mat& input, cv::Mat& output)
{

	cout << "Input image step: " << input.step << " rows: " << input.rows << " cols: " << input.cols << endl;

	size_t colorBytes = input.step * input.rows;
	size_t grayBytes = output.step * output.rows;
	size_t tempSize = 256 *sizeof(int);

	int *temp;
	unsigned char *d_input, *d_output;

	// Allocate device memory
	SAFE_CALL(cudaMalloc(&d_input, colorBytes), "CUDA Malloc Failed");
	SAFE_CALL(cudaMalloc(&d_output, grayBytes), "CUDA Malloc Failed");
	SAFE_CALL(cudaMalloc(&temp, tempSize), "CUDA Malloc Failed");

	// Copy data from OpenCV input image to device memory
	SAFE_CALL(cudaMemcpy(d_input, input.ptr(), colorBytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");
	SAFE_CALL(cudaMemcpy(d_output, output.ptr(), colorBytes, cudaMemcpyHostToDevice), "CUDA Memcpy Host To Device Failed");

	// Specify a reasonable block size
	const dim3 block(16, 16);

	// Calculate grid size to cover the whole image
	const dim3 grid((int)ceil((float)input.cols / block.x), (int)ceil((float)input.rows/ block.y));
	printf("equalization_kernel<<<(%d, %d) , (%d, %d)>>>\n", grid.x, grid.y, block.x, block.y);

	// Launch the color conversion kernel
	auto start_cpu =  chrono::high_resolution_clock::now();
	
	generate_histogram <<<grid, block >>>(d_input, d_output, input.cols, input.rows, static_cast<int>(input.step),temp);
	equalization_phase1 <<<grid, block >>>(d_input, d_output, temp);
	equalization_phase2 <<<grid, block >>>(d_input, d_output, input.cols * input.rows,temp);
	save_to_image <<<grid, block >>>(d_input, d_output, input.cols, input.rows, static_cast<int>(input.step),temp);
	
	auto end_cpu =  chrono::high_resolution_clock::now();
	chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;
	printf("elapsed %f ms\n", duration_ms.count());

	// Synchronize to check for any kernel launch errors
	SAFE_CALL(cudaDeviceSynchronize(), "Kernel Launch Failed");

	// Copy back data from destination device meory to OpenCV output image
	SAFE_CALL(cudaMemcpy(output.ptr(), d_output, grayBytes, cudaMemcpyDeviceToHost), "CUDA Memcpy Host To Device Failed");

	// Free the device memory
	SAFE_CALL(cudaFree(d_input), "CUDA Free Failed");
	SAFE_CALL(cudaFree(d_output), "CUDA Free Failed");
}

int main(int argc, char *argv[])
{
	string imagePath;

	if(argc < 2)
		imagePath = "Images/dog1.jpeg";
  	else
  		imagePath = argv[1];

	// Read input image from the disk
	cv::Mat input = cv::imread(imagePath, CV_LOAD_IMAGE_COLOR);

	if (input.empty())
	{
		cout << "Image Not Found!" << std::endl;
		cin.get();
		return -1;
	}

	cv::Mat input_bw(input.rows, input.cols, CV_8UC1);
	cv::Mat output(input.rows, input.cols, CV_8UC1);

	cv::cvtColor(input, input_bw, cv::COLOR_BGR2GRAY);

	
	equalization(input_bw, output);

	//Allow the windows to resize
	namedWindow("Input", cv::WINDOW_NORMAL);
	namedWindow("Output", cv::WINDOW_NORMAL);

	cv::resizeWindow("Input", 800, 600);
	cv::resizeWindow("Output", 800, 600);

	//Show the input and output
	imshow("Input", input_bw);
	imshow("Output", output);

	//Wait for key press
	cv::waitKey();

	return 0;
}
