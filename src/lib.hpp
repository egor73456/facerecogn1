#pragma once

#include <atomic>
#include <chrono>
#include <future>
#include <iostream>
#include <mutex>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <string>

extern const double inScaleFactor;
extern const cv::Scalar meanVal;
extern const float confidenceThreshold;

extern std::mutex im_access, main_im_access;

extern std::string caffeConfigFile;
extern std::string caffeWeightFile;

extern std::chrono::_V2::system_clock::time_point prev_time, curt_time;
extern int fps, fps_max;
extern cv::Scalar draw_color;

struct BreakThroughContainer {
  cv::Mat &draw, &frame;
  std::unique_ptr<cv::Point> prev;
  int drawing;
};

extern void detectFaceOpenCVDNN(cv::dnn::Net &net, cv::Mat &frameOpenCVDNN,
                                cv::Mat &ret, std::atomic<bool> &stop);
extern void process_im(cv::Mat &frame, int &hue, int &sat, int &val,
                       int &inverse_colors, cv::Mat &drawing_layer);
extern void mouseHandler(int event, int x, int y, __attribute__((unused)) int flags,
                  void *param);