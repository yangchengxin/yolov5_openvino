#pragma once
#ifndef CPPDLL_EXPORTS
#define CPPDLL_API __declspec(dllexport)
#else
#define CPPDLL_API __declspec(dllimport)
#endif
#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include <thread>
#include <fstream>
#include <iostream>
#include <onnxruntime_cxx_api.h>
#include "tinyxml2.h"
#include <crtdbg.h>
#include <time.h>
#include <chrono>
#include <mutex>
#include "Logger.h"
#include "calibrator.h"

using namespace std;
using namespace cv;
using namespace ov;
using namespace std::chrono;
//std::mutex mtx;

struct SliceImageData {
	//输入的原图
	cv::Mat Original_Frame;

	//分图之后的Mat数组
	vector<cv::Mat> Slice_Batch;

	//分图的行与列
	int Slice_Cols;
	int Slice_Rows;

	//没有做overlap时的各分图的宽与高
	int Slice_width;
	int Slice_height;

	//分图时宽与高的overlap
	int Slice_Overlap_W;
	int Slice_Overlap_H;
};

SliceImageData Slice_Image(cv::Mat frame, const int col_numbers, const int row_numbers, const int w_overlap, const int h_overlap);
extern "C" CPPDLL_API void Initial_model();
extern "C" CPPDLL_API int Detect(uchar* input_ptr, int height, int width, int channels, int col_nums, int row_nums, int w_overlap, int h_overlap, char*& output);