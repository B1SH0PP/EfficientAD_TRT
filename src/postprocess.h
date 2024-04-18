#pragma once

#include <opencv2/opencv.hpp>

void genHeatMap(cv::Mat originImg, cv::Mat& anomalyGrayMap, cv::Mat& HeatMap);