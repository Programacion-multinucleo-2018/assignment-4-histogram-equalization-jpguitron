#include <iostream>
#include <cstdio>
#include <cmath>
#include <chrono>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
//g++ -o exe img_eq.cpp -lopencv_core -lopencv_highgui -lopencv_imgproc -std=c++11

void equalization(const cv::Mat& input, cv::Mat& output)
{
  cout << "Input image step: " << input.step << " rows: " << input.rows << " cols: " << input.cols << endl;

	float imgSize = input.rows * input.cols;

	int h[256] = {};
	int h_s[256] = {};
	unsigned int actual;

  for (int y = 0; y < input.rows; y++)
  {
    for (int x = 0; x < input.cols; x++)
    {
			actual = (int)input.at<uchar>(y,x);
			h[actual]++;
    }
  }
	for(int y = 0; y < 256;y++)
	{
		for(int x = 0; x <= y; x++)
		{ 
			h_s[y] += h[x];
		}
	}

	for(int y = 0; y < 256;y++)
		h_s[y] = h_s[y]*(255/imgSize); 

	for (int y = 0; y < input.rows; y++)
  {
    for (int x = 0; x < input.cols; x++)
    {
			actual = (int)input.at<uchar>(y,x);
			output.at<uchar>(y,x) = h_s[actual];
    }
  }

}

int main(int argc, char *argv[])
{
	string imagePath;

	if(argc < 2)
		imagePath = "Images/dog1.jpeg";
  else
  	imagePath = argv[1];

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

	output = input_bw.clone();


	auto start_cpu =  chrono::high_resolution_clock::now();
	equalization(input_bw, output);
	auto end_cpu =  chrono::high_resolution_clock::now();
	chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;
	printf("elapsed %f ms\n", duration_ms.count());

	//Allow the windows to resize
	namedWindow("Input", cv::WINDOW_NORMAL);
	namedWindow("Output", cv::WINDOW_NORMAL);

	//Show the input and output
	imshow("Input", input_bw);
	imshow("Output", output);

	//Wait for key press
	cv::waitKey();

	return 0;
}
