#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include <stdio.h>

#ifndef CPPDLL_EXPORTS
#define CPPDLL_API __declspec(dllexport)
#else
#define CPPDLL_API __declspec(dllimport)
#endif

using namespace std;

const std::string camera_config = "./CameraCalibrator/";
const std::string yaml_name = "CalibrationParams.yaml";

extern "C" CPPDLL_API void Calibrate(uchar * input_image, int rows, int cols, uchar* output_image);